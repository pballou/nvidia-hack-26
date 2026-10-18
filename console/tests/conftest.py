"""Live servers for the console tests: the real sandbox app and console apps, each uvicorn in a thread on a free port."""

from __future__ import annotations

import json
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from console import asr
from console.app import ConsoleSettings, create_app

# Voice input in tests: the stub engine, chosen explicitly. The console's default is Parakeet, and the tests
# must never call the shared sidecar on :8096 (whatever CONSOLE_ASR says on this machine). This is StubEngine
# itself, the recognizer stand-in whose text the page treats like speech, not `--asr stub` (FixedTextEngine,
# "stub (fixed text)", which never auto-runs). Every asr.set_engine(None) falls back to it.
TEST_ASR = "test-stub"
asr.ENGINES[TEST_ASR] = asr.StubEngine
asr.configure(TEST_ASR)


def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: headless Chromium with the real mock agent (12-task console run, "
                                       "keyboard-only UI tests); in the default suite, skip with -m 'not e2e'")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerThread:
    def __init__(self, app, port: int | None = None):
        self.port = port or free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning",
                                                    timeout_graceful_shutdown=1))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> str:
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started:
            if time.time() > deadline or not self.thread.is_alive():
                raise RuntimeError("server did not start")
            time.sleep(0.02)
        return self.url

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


@pytest.fixture(scope="session")
def sandbox_url():
    from sandbox.app import app

    srv = ServerThread(app)
    yield srv.start()
    srv.stop()


@pytest.fixture
def console(sandbox_url, tmp_path):
    """make(fake=(run_task, make_cfg) | None, **settings) -> (base_url, app). fake=None uses --agent's agent."""
    servers: list[ServerThread] = []

    def make(fake=None, **kw):
        kw.setdefault("sandbox_url", sandbox_url)
        kw.setdefault("runs_dir", tmp_path / "runs")
        kw.setdefault("static_dir", tmp_path / "static")
        kw.setdefault("stop_timeout_s", 5.0)
        app = create_app(ConsoleSettings(**kw), agent=fake)
        srv = ServerThread(app)
        servers.append(srv)
        return srv.start(), app

    yield make
    for srv in servers:
        srv.stop()


def read_sse(url: str, headers: dict | None = None, timeout: float = 30.0) -> list[dict]:
    """Read an SSE stream until `end` (or close). Messages: {id, event, data}; comments: {comment}."""
    out: list[dict] = []
    cur: dict = {}
    with httpx.stream("GET", url, headers=headers, timeout=timeout) as r:
        assert r.status_code == 200, r.status_code
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if line == "":
                if cur:
                    out.append(cur)
                    if cur.get("event") == "end":
                        break
                cur = {}
            elif line.startswith(":"):
                out.append({"comment": line[1:].strip()})
            else:
                key, _, value = line.partition(":")
                value = value[1:] if value.startswith(" ") else value
                cur[key] = int(value) if key == "id" else json.loads(value) if key == "data" else value
    return out


def messages(stream: list[dict]) -> list[dict]:
    return [m for m in stream if "event" in m]
