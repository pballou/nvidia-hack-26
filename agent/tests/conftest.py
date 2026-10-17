"""Shared fixtures for agent tests: the fake sandbox and real headless Chromium sessions.

Browser fixtures are function-scoped because pytest-asyncio gives every test its own
event loop and Playwright objects are bound to the loop that made them.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator

import pytest
import pytest_asyncio

from agent.browser import BrowserPool, BrowserSession, close_pool, get_pool
from agent.config import BrowserConfig, Viewport
from agent.testing.fake_sandbox import FakeSandboxServer


@pytest.fixture(scope="session")
def fake_sandbox_url() -> Iterator[str]:
    """Base URL of one FakeSandboxServer for the whole test run; isolate tests with the `sid` fixture."""
    with FakeSandboxServer() as url:
        yield url


@pytest.fixture
def sid() -> str:
    """A fresh sandbox session id."""
    return "t-" + uuid.uuid4().hex[:10]


@pytest.fixture
def browser_cfg() -> BrowserConfig:
    return BrowserConfig(headless=True)


@pytest_asyncio.fixture
async def browser_pool(browser_cfg: BrowserConfig) -> AsyncIterator[BrowserPool]:
    """The process pool for this test's loop, closed afterwards so no Chromium outlives the loop."""
    pool = await get_pool(browser_cfg)
    yield pool
    await close_pool()


@pytest_asyncio.fixture
async def browser_session(browser_pool: BrowserPool) -> AsyncIterator[BrowserSession]:
    """A fresh 1280x800 context. Navigate it yourself (`await browser_session.goto(...)`)."""
    session = await browser_pool.new_session(Viewport())
    yield session
    await session.close()
