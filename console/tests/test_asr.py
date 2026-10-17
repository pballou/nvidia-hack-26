"""console/asr.py: WAV validation, engine selection, and the Parakeet engine against a fake sidecar.

The fake speaks NeMo-Speech.cpp's protocol (GET /health, POST /v1/audio/transcriptions with
multipart file + response_format=json), so these tests need no model.
"""

from __future__ import annotations

import io
import math
import socket
import struct
import threading
import time
import wave

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from console import asr
from console.app import ConsoleSettings, create_app


def make_wav(seconds: float = 0.5, rate: int = 16_000, channels: int = 1, width: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        n = int(seconds * rate)
        if width == 2:
            w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(i / 8))) * channels for i in range(n)))
        else:
            w.writeframes(bytes([128]) * n * channels)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def stub_engine_after():
    yield
    asr.configure("stub")


class FakeSidecar:
    def __init__(self):
        self.calls: list[dict] = []
        self.reply = (200, {"text": "  Turn on high contrast.  "})
        app = FastAPI()

        @app.get("/health")
        def health():
            return {"status": "ok", "version": "fake"}

        @app.post("/v1/audio/transcriptions")
        async def transcriptions(request: Request):
            form = await request.form()
            upload = form["file"]
            self.calls.append({"response_format": form.get("response_format"), "filename": upload.filename,
                               "wav": await upload.read()})
            status, body = self.reply
            return JSONResponse(body, status_code=status)

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        for _ in range(200):
            try:
                if httpx.get(f"{self.url}/health", timeout=0.5, trust_env=False).status_code == 200:
                    return self
            except httpx.HTTPError:
                time.sleep(0.05)
        raise RuntimeError("fake sidecar did not start")

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=5)


@pytest.fixture
def sidecar():
    with FakeSidecar() as fake:
        yield fake


def test_read_wav_accepts_16k_mono_pcm16():
    frames, duration = asr.read_wav(make_wav(0.5))
    assert len(frames) == 8000 * 2 and duration == pytest.approx(0.5)


@pytest.mark.parametrize("data,match", [
    (b"", "No audio"),
    (b"not a wav at all, just text", "must be a WAV"),
    (make_wav(channels=2), "mono"),
    (make_wav(rate=44_100), "16 kHz"),
    (make_wav(width=1), "16-bit"),
    (make_wav(0.1), "too short"),
    (make_wav(61), "too long"),
    (make_wav(0.5)[:30], "WAV"),
])
def test_read_wav_rejects(data, match):
    with pytest.raises(asr.AudioError, match=match):
        asr.read_wav(data)


def test_stub_is_the_default_engine():
    assert asr.info() == {"available": True, "model": "stub", "label": "stub"}
    out = asr.transcribe(make_wav())
    assert out["text"] == "Turn on high contrast." and out["model"] == "stub" and out["duration_s"] == 0.5


def test_parakeet_engine_speaks_the_sidecar_protocol(sidecar, monkeypatch):
    monkeypatch.setenv("CONSOLE_ASR_URL", sidecar.url)
    asr.configure("parakeet")
    assert asr.info() == {"available": True, "model": "nvidia/parakeet-tdt-0.6b-v3", "label": "Parakeet TDT 0.6B v3"}
    audio = make_wav(0.75)
    out = asr.transcribe(audio)
    assert out["text"] == "Turn on high contrast." and out["model"] == "nvidia/parakeet-tdt-0.6b-v3"
    assert out["duration_s"] == 0.75 and out["latency_ms"] >= 0
    (call,) = sidecar.calls
    assert call["response_format"] == "json" and call["filename"].endswith(".wav")
    with wave.open(io.BytesIO(call["wav"])) as w:  # re-wrapped as 16 kHz mono PCM16, same samples
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16_000)
        assert w.readframes(w.getnframes()) == asr.read_wav(audio)[0]


def test_parakeet_errors_map_to_audio_and_engine_errors(sidecar):
    engine = asr.NemoSpeechEngine(sidecar.url)
    asr.set_engine(engine)
    sidecar.reply = (400, {"error": "bad audio"})
    with pytest.raises(asr.AudioError, match="rejected"):
        asr.transcribe(make_wav())
    sidecar.reply = (500, {"error": "boom"})
    with pytest.raises(asr.EngineUnavailable, match="500"):
        asr.transcribe(make_wav())


def test_parakeet_unreachable_is_reported_not_raised(monkeypatch):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead = f"http://127.0.0.1:{s.getsockname()[1]}"
    monkeypatch.setenv("CONSOLE_ASR_URL", dead)
    asr.configure("parakeet")
    info = asr.info()
    assert info["available"] is False and "not reachable" in info["error"]
    with pytest.raises(asr.EngineUnavailable):
        asr.transcribe(make_wav())


def test_transcribe_endpoint_with_parakeet_and_when_it_is_down(sidecar, monkeypatch):
    monkeypatch.setenv("CONSOLE_ASR_URL", sidecar.url)
    asr.configure("parakeet")
    client = TestClient(create_app(ConsoleSettings(agent="mock")))
    assert client.get("/api/config").json()["asr"]["label"] == "Parakeet TDT 0.6B v3"
    r = client.post("/api/transcribe", content=make_wav(), headers={"content-type": "audio/wav"})
    assert r.status_code == 200 and r.json()["text"] == "Turn on high contrast."
    r = client.post("/api/transcribe", files={"audio": ("a.wav", make_wav(1.0), "audio/wav")})
    assert r.status_code == 200 and r.json()["duration_s"] == 1.0
    assert client.post("/api/transcribe", content=make_wav(channels=2),
                       headers={"content-type": "audio/wav"}).status_code == 400
    sidecar.reply = (503, {"error": "busy"})
    r = client.post("/api/transcribe", content=make_wav(), headers={"content-type": "audio/wav"})
    assert r.status_code == 503 and "Parakeet error 503" in r.json()["error"]
