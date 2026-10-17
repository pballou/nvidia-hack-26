"""Shared fixtures: a live sandbox server (uvicorn in a thread) and a Playwright browser.

Browser contexts match the agent's: 1280x800, device_scale_factor=1, reduced motion,
fixed locale/timezone. DOM selectors are allowed here (tests/oracle only).
"""

from __future__ import annotations

import itertools
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from sandbox.app import TASKS_BY_ID, app

VIEWPORT = {"width": 1280, "height": 800}
CONTEXT_OPTS = dict(viewport=VIEWPORT, device_scale_factor=1, reduced_motion="reduce",
                    locale="en-US", timezone_id="America/New_York")
_sid_counter = itertools.count(1)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def sandbox_url():
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(200):
        try:
            if httpx.get(f"{url}/api/health", timeout=0.5).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    else:
        raise RuntimeError("sandbox server did not start")
    yield url
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="session")
def browser():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


def with_sid(url: str, sid: str) -> str:
    return f"{url}{'&' if '?' in url else '?'}sid={sid}"


@pytest.fixture
def reset(sandbox_url):
    """reset(task_id=None, sid=None) -> sid, via POST /api/reset."""

    def _reset(task_id: str | None = None, sid: str | None = None) -> str:
        sid = sid or f"test-{next(_sid_counter)}"
        r = httpx.post(f"{sandbox_url}/api/reset", json={"sid": sid, "task_id": task_id}, timeout=5)
        r.raise_for_status()
        return sid

    return _reset


@pytest.fixture
def get_state(sandbox_url):
    def _get(sid: str) -> dict:
        r = httpx.get(f"{sandbox_url}/api/state", params={"sid": sid}, timeout=5)
        r.raise_for_status()
        return r.json()

    return _get


@pytest.fixture
def open_page(browser, sandbox_url, reset):
    """open_page(task_id=None, path=None, sid=None) -> (page, sid).

    Fresh context, session reset to the task's initial state, page at the task's
    start_path (or `path`). Contexts are closed after the test.
    """
    contexts = []

    def _open(task_id: str | None = None, path: str | None = None, sid: str | None = None):
        sid = reset(task_id, sid)
        ctx = browser.new_context(**CONTEXT_OPTS)
        contexts.append(ctx)
        page = ctx.new_page()
        start = path or (TASKS_BY_ID[task_id]["start_path"] if task_id else "/")
        page.goto(with_sid(f"{sandbox_url}{start}", sid))
        page.wait_for_load_state("networkidle")
        return page, sid

    yield _open
    for ctx in contexts:
        ctx.close()
