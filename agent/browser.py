"""Async Playwright (1.62.0) wrapper: one Chromium per process (per headless value), one context per agent session.

Vision-only (AGENTS.md, contracts.md §6): nothing here queries the DOM or the app state.
The page is seen only through `screenshot()` and driven only through the mouse and
keyboard; `url()` and `title()` are browser chrome. Contexts are pinned for reproducible
pixels: viewport 1280x800 CSS px at device_scale_factor 1, reduced motion, light color
scheme, fixed locale/timezone, sRGB colors.

Per session:
- One page. A popup or target=_blank tab becomes the session's page and the previous
  page is closed, like a browser that opens every link in the same window.
- JS dialogs (alert/confirm/prompt/beforeunload) are accepted automatically and logged in
  `BrowserSession.dialogs` (they never appear in screenshots).
- After every click or scroll the mouse is parked just outside the viewport (see
  `BrowserSession._park`), so hover styles never differ between screenshots. Menus that
  open on hover alone close again; the agent has to click them.

Usage:
    pool = await get_pool(cfg.browser)
    session = await pool.new_session(cfg.viewport, navigation_timeout_s=cfg.timeouts.navigation_s)
    await session.goto(start_url)   # harness only; the agent acts via click/type/press/scroll
    ...
    await session.close()
"""

from __future__ import annotations

import asyncio
import io
import math

from PIL import Image, ImageChops
from playwright.async_api import Browser, BrowserContext, Dialog, Page, Playwright, async_playwright
from playwright.async_api import Error as PlaywrightError

from agent.config import BrowserConfig, Timeouts, Viewport

SMALL_SCROLL_PX = 240
PAGE_SCROLL_FRACTION = 0.85    # of the viewport height, so consecutive pages overlap a little
TYPE_DELAY_MS = 15             # per character; lets key handlers on the page keep up
# Screenshot colors must not depend on the host display's color profile (headed runs on a Mac).
CHROMIUM_ARGS = ("--force-color-profile=srgb",)


def screen_changed(before: Image.Image, after: Image.Image, *, size: tuple[int, int] = (640, 400),
                   pixel_delta: int = 24, min_pixels: int = 2) -> bool:
    """True if the screen visibly changed between two screenshots. Different sizes count as changed.

    Both are downsampled to `size` grayscale (BOX filter) and compared cell by cell: changed
    if at least `min_pixels` cells differ by more than `pixel_delta` (0-255). At 1280x800 a
    cell is 2x2 px. Measured in Chromium: two idle screenshots differ in 0 cells (caret hidden,
    animations off), one typed "x" in a default-font input moves ~10 cells and a "." 2, and a
    toggle flip hundreds, while a single changed pixel touches only one cell.
    """
    if before.size != after.size:
        return True
    a = before.convert("L").resize(size, Image.Resampling.BOX)
    b = after.convert("L").resize(size, Image.Resampling.BOX)
    hist = ImageChops.difference(a, b).histogram()
    return sum(hist[pixel_delta + 1:]) >= min_pixels


class BrowserSession:
    """One isolated browser context (own cookies and storage) with a single page.

    Implements the Browser protocol of agent/session.py.
    """

    def __init__(self, context: BrowserContext, page: Page, viewport: Viewport, cfg: BrowserConfig):
        self.context = context
        # Exposed ONLY for agent/testing/oracle_grounder.py (the test-only DOM oracle) and tests.
        # Agent code must never use it: it sees the page through screenshot() alone.
        self.page = page
        self.viewport = viewport
        self.cfg = cfg
        self.dialogs: list[dict] = []    # {type, message} of every auto-accepted JS dialog
        self._closed = False
        context.on("dialog", self._on_dialog)
        context.on("page", self._on_page)

    async def _on_dialog(self, dialog: Dialog) -> None:
        self.dialogs.append({"type": dialog.type, "message": dialog.message})
        try:
            await dialog.accept()
        except PlaywrightError:  # page closed meanwhile
            pass

    async def _on_page(self, page: Page) -> None:
        """A popup / new tab replaces the current page, which is closed (one page per session)."""
        if page is self.page or self._closed:
            return
        old, self.page = self.page, page
        try:
            await old.close()
        except PlaywrightError:
            pass

    async def goto(self, url: str) -> None:
        """Harness only (start URL, Gym seed_session); not an agent tool. Raises if the page can't load."""
        await self.page.goto(url, wait_until="domcontentloaded")  # context default timeout = navigation_s
        await self._park()  # same mouse position as after any later action
        await self.settle()

    async def screenshot(self) -> Image.Image:
        """The viewport as an RGB image of exactly viewport size (CSS px = device px)."""
        png = await self.page.screenshot(type="png", animations="disabled", caret="hide", scale="css")
        img = Image.open(io.BytesIO(png)).convert("RGB")
        size = (self.viewport.width, self.viewport.height)
        return img if img.size == size else img.resize(size, Image.Resampling.BILINEAR)

    async def click(self, x: float, y: float) -> None:
        """Left-click at viewport CSS px (clamped into the viewport), then park the mouse."""
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"click point must be finite, got ({x}, {y})")
        x = min(max(float(x), 0.0), self.viewport.width - 1)
        y = min(max(float(y), 0.0), self.viewport.height - 1)
        await self.page.mouse.click(x, y)
        await self._park()

    async def type(self, text: str) -> None:
        await self.page.keyboard.type(text, delay=TYPE_DELAY_MS)

    async def press(self, key: str) -> None:
        """A Playwright key name or chord ('Enter', 'Control+a'); unknown names raise."""
        await self.page.keyboard.press(key)

    async def select_all(self) -> None:
        await self.page.keyboard.press("ControlOrMeta+A")

    async def scroll(self, direction: str, amount: str) -> None:
        """Mouse wheel at the viewport center: 'small' = 240 px, 'page' = 85% of the viewport height."""
        if direction not in ("up", "down"):
            raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")
        if amount not in ("small", "page"):
            raise ValueError(f"amount must be 'small' or 'page', got {amount!r}")
        dy = SMALL_SCROLL_PX if amount == "small" else round(self.viewport.height * PAGE_SCROLL_FRACTION)
        await self.page.mouse.move(self.viewport.width / 2, self.viewport.height / 2)
        await self.page.mouse.wheel(0, dy if direction == "down" else -dy)
        await self._park()

    async def settle(self) -> None:
        """Let the page react to the last action; never raises.

        Network idle (capped at network_idle_ms), a fixed settle_ms pause, then network idle
        again for a navigation that committed during the pause (an idle page returns at once).
        """
        await self._wait_idle()
        await asyncio.sleep(self.cfg.settle_ms / 1000)
        await self._wait_idle()

    async def _wait_idle(self) -> None:
        for _ in range(2):  # once more if a popup replaced the page while waiting
            page = self.page
            try:
                await page.wait_for_load_state("networkidle", timeout=self.cfg.network_idle_ms)
            except PlaywrightError:  # timeout (TimeoutError is a subclass) or page closed
                pass
            if self.page is page:
                return

    async def _park(self) -> None:
        """Move the mouse to (width, height): one px past the bottom-right corner, outside the viewport.

        Nothing is hit-tested there, so no element keeps a :hover style (checked in Chromium;
        an in-viewport corner would hover whatever sits there, e.g. a toast). Unlike the top
        edge it doesn't trip "exit intent" popups.
        """
        try:
            await self.page.mouse.move(self.viewport.width, self.viewport.height)
        except PlaywrightError:  # page closed or replaced by a popup mid-call
            pass

    async def url(self) -> str:
        try:
            return self.page.url
        except PlaywrightError:
            return ""

    async def title(self) -> str:
        try:
            return await self.page.title()
        except PlaywrightError:  # mid-navigation or closed
            return ""

    def screen_changed(self, before: Image.Image, after: Image.Image) -> bool:
        return screen_changed(before, after)

    async def close(self) -> None:
        """Close the context (and its page). Idempotent."""
        if self._closed:
            return
        self._closed = True
        try:
            await self.context.close()
        except PlaywrightError:  # browser already gone
            pass


class BrowserPool:
    """One Chromium; `new_session()` gives each agent run its own context.

    `cfg.headless` is fixed at launch. The other BrowserConfig fields (locale, timezone, settle)
    are per session: `cfg` here is only the default for sessions that don't pass their own.
    """

    def __init__(self, cfg: BrowserConfig):
        self.cfg = cfg
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self._lock = asyncio.Lock()  # two sessions starting after a crash must not both launch

    @property
    def connected(self) -> bool:
        return self._browser is not None and self._browser.is_connected()

    async def start(self) -> None:
        async with self._lock:
            if self.connected:
                return
            await self.close()  # a disconnected browser's driver
            self._pw = await async_playwright().start()
            self._browser = await self._pw.chromium.launch(headless=self.cfg.headless, args=list(CHROMIUM_ARGS))

    async def new_session(self, viewport: Viewport, *, navigation_timeout_s: float = Timeouts.navigation_s,
                          cfg: BrowserConfig | None = None) -> BrowserSession:
        cfg = cfg or self.cfg
        if not self.connected:
            await self.start()
        size = {"width": viewport.width, "height": viewport.height}
        context = await self._browser.new_context(
            viewport=size, screen=size, device_scale_factor=1, reduced_motion="reduce", color_scheme="light",
            locale=cfg.locale, timezone_id=cfg.timezone_id)
        context.set_default_timeout(navigation_timeout_s * 1000)  # navigation and screenshots
        page = await context.new_page()
        return BrowserSession(context, page, viewport, cfg)

    async def close(self) -> None:
        """Close the browser (and every context) and stop Playwright. Idempotent."""
        browser, pw, self._browser, self._pw = self._browser, self._pw, None, None
        if browser is not None:
            try:
                await browser.close()
            except PlaywrightError:
                pass
        if pw is not None:
            try:
                await pw.stop()
            except Exception:  # driver already exited
                pass


class _ConfiguredPool:
    """A shared BrowserPool whose sessions use another caller's per-session settings (see get_pool)."""

    def __init__(self, pool: BrowserPool, cfg: BrowserConfig):
        self.pool, self.cfg = pool, cfg

    @property
    def connected(self) -> bool:
        return self.pool.connected

    async def new_session(self, viewport: Viewport, *,
                          navigation_timeout_s: float = Timeouts.navigation_s) -> BrowserSession:
        return await self.pool.new_session(viewport, navigation_timeout_s=navigation_timeout_s, cfg=self.cfg)


# Process-wide pools (one per headless value), bound to the event loop that created them
# (Playwright objects are loop-bound).
_pools: dict[bool, BrowserPool] = {}
_pool_loop: asyncio.AbstractEventLoop | None = None
_pool_lock: asyncio.Lock | None = None


async def get_pool(cfg: BrowserConfig) -> BrowserPool | _ConfiguredPool:
    """The process-wide Chromium for the running event loop and `cfg.headless`, started on first use.

    Its sessions get `cfg`'s per-session settings. A pool is replaced only when its browser has
    disconnected, never because another caller's settings differ (that would kill live runs).
    A pool from another loop (pytest-asyncio runs every test in a new loop) can't be used or
    closed from here, so it is dropped: call close_pool() before a loop ends to avoid leaking
    its Chromium.
    """
    global _pools, _pool_loop, _pool_lock
    loop = asyncio.get_running_loop()
    if _pool_loop is not loop:
        _pools, _pool_loop, _pool_lock = {}, loop, asyncio.Lock()
    async with _pool_lock:
        pool = _pools.get(cfg.headless)
        if pool is None or not pool.connected:
            if pool is not None:
                await pool.close()
            pool = BrowserPool(cfg)
            await pool.start()
            _pools[cfg.headless] = pool
    return pool if pool.cfg == cfg else _ConfiguredPool(pool, cfg)


async def close_pool() -> None:
    """Close the running loop's pools, if any. Safe to call repeatedly."""
    global _pools
    if _pools and _pool_loop is asyncio.get_running_loop():
        pools, _pools = list(_pools.values()), {}
        for pool in pools:
            await pool.close()
