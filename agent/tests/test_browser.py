"""agent/browser.py: screen_changed on synthetic images, then a real headless Chromium against the fake sandbox.

Tests may read the DOM (bounding boxes, input values, scroll offsets) and /api/state to
aim and to check results; the agent itself never does.
"""

from __future__ import annotations

import asyncio
from urllib.parse import quote

import httpx
import pytest
from PIL import Image, ImageDraw
from playwright.async_api import Error as PlaywrightError

from agent.browser import BrowserSession, close_pool, get_pool, screen_changed
from agent.config import BrowserConfig, Viewport

W, H = 1280, 800
BACKGROUND = (1100, 650)   # empty page background on the fake sandbox pages (cards end at x~690)


def _state(base_url: str, sid: str) -> dict:
    return httpx.get(f"{base_url}/api/state", params={"sid": sid}, timeout=5).json()


async def _center(session: BrowserSession, selector: str) -> tuple[float, float]:
    box = await session.page.locator(selector).bounding_box()
    assert box, f"{selector} not visible"
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


def _data_url(html: str) -> str:
    return "data:text/html;charset=utf-8," + quote(html)


# --- screen_changed (synthetic) ------------------------------------------------------

def _screen() -> Image.Image:
    img = Image.new("RGB", (W, H), (244, 246, 248))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 64], fill=(11, 79, 138))
    d.rectangle([48, 120, 688, 420], fill=(255, 255, 255), outline=(208, 215, 222))
    d.text((72, 150), "High contrast", fill=(29, 39, 51))
    return img


def _toggle(img: Image.Image, on: bool, knob_right: bool | None = None, x: int = 600, y: int = 145) -> Image.Image:
    """A 56x30 switch like the sandbox's: gray/knob-left when off, green/knob-right when on."""
    img = img.copy()
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([x, y, x + 55, y + 29], radius=15, fill=(26, 127, 55) if on else (204, 204, 204),
                        outline=(85, 85, 85), width=2)
    kx = x + 29 if (on if knob_right is None else knob_right) else x + 5
    d.ellipse([kx, y + 5, kx + 20, y + 25], fill=(255, 255, 255))
    return img


def test_identical_screens_are_unchanged():
    img = _toggle(_screen(), on=False)
    assert not screen_changed(img, img)
    assert not screen_changed(img, img.copy())


def test_small_toggle_flip_is_a_change():
    base = _screen()
    assert screen_changed(_toggle(base, on=False), _toggle(base, on=True))


def test_knob_move_alone_is_a_change():
    base = _screen()
    assert screen_changed(_toggle(base, on=False), _toggle(base, on=False, knob_right=True))


def test_noise_is_not_a_change():
    img = _toggle(_screen(), on=False)
    one_pixel = img.copy()
    one_pixel.putpixel((900, 500), (0, 0, 0))
    brighter = img.point(lambda v: min(255, v + 8))
    assert not screen_changed(img, one_pixel)
    assert not screen_changed(img, brighter)


def test_different_sizes_are_a_change():
    assert screen_changed(_screen(), _screen().resize((W // 2, H // 2)))


# --- real Chromium + fake sandbox --------------------------------------------------------

@pytest.mark.asyncio
async def test_screenshot_is_viewport_sized_rgb(browser_session, fake_sandbox_url, sid):
    await browser_session.goto(f"{fake_sandbox_url}/settings?sid={sid}")
    img = await browser_session.screenshot()
    assert img.size == (W, H) and img.mode == "RGB"
    assert browser_session.viewport == Viewport(W, H)
    assert await browser_session.title() == "Settings - Acme Utilities"
    assert (await browser_session.url()).startswith(f"{fake_sandbox_url}/settings")


@pytest.mark.asyncio
async def test_clicking_the_toggle_changes_screen_and_state(browser_session, fake_sandbox_url, sid):
    await browser_session.goto(f"{fake_sandbox_url}/settings?sid={sid}")
    before = await browser_session.screenshot()
    await browser_session.click(*await _center(browser_session, "#hc"))
    await browser_session.settle()
    after = await browser_session.screenshot()
    assert browser_session.screen_changed(before, after)
    assert _state(fake_sandbox_url, sid)["settings"]["accessibility"]["high_contrast"] is True


@pytest.mark.asyncio
async def test_clicking_empty_background_changes_nothing(browser_session, fake_sandbox_url, sid):
    await browser_session.goto(f"{fake_sandbox_url}/settings?sid={sid}")
    before = await browser_session.screenshot()
    assert not screen_changed(before, await browser_session.screenshot())  # screenshots are stable
    await browser_session.click(*BACKGROUND)
    await browser_session.settle()
    assert not browser_session.screen_changed(before, await browser_session.screenshot())
    assert _state(fake_sandbox_url, sid)["settings"]["accessibility"]["high_contrast"] is False


@pytest.mark.asyncio
async def test_small_switch_flip_is_detected_in_the_browser(browser_session):
    """Only a 56x30 switch changes (no page-wide restyle as with the sandbox's high contrast)."""
    await browser_session.goto(_data_url(
        "<style>body{margin:40px;background:#f4f6f8} .sw{width:56px;height:30px;border-radius:15px;"
        "border:2px solid #555;background:#ccc;position:relative;box-sizing:border-box}"
        ".sw::after{content:'';position:absolute;top:3px;left:3px;width:20px;height:20px;border-radius:10px;"
        "background:#fff} .sw.on{background:#1a7f37} .sw.on::after{left:29px}</style>"
        "<p>Captions</p><div id=sw class=sw onclick=\"this.classList.toggle('on')\"></div>"))
    before = await browser_session.screenshot()
    await browser_session.click(*await _center(browser_session, "#sw"))
    await browser_session.settle()
    assert await browser_session.page.locator("#sw.on").count() == 1
    assert browser_session.screen_changed(before, await browser_session.screenshot())


@pytest.mark.asyncio
async def test_one_character_edit_is_a_change_and_idle_is_not(browser_session):
    await browser_session.goto(_data_url("<body style='margin:40px'><input id=f value='Sam Rivera12'></body>"))
    x, y = await _center(browser_session, "#f")
    await browser_session.click(x, y)
    await browser_session.press("End")
    idle = await browser_session.screenshot()
    await asyncio.sleep(0.3)
    a = await browser_session.screenshot()
    assert not screen_changed(idle, a)
    await browser_session.type("x")
    b = await browser_session.screenshot()
    await browser_session.press("Backspace")
    await browser_session.press("Backspace")
    c = await browser_session.screenshot()  # "x" and then the "2" deleted
    await browser_session.type("2")
    d = await browser_session.screenshot()
    assert screen_changed(a, b) and screen_changed(b, c) and screen_changed(c, d)


@pytest.mark.asyncio
async def test_hover_style_alone_is_not_a_change(browser_session):
    """The mouse is parked off-page after a click, so a :hover-only style doesn't fake a change."""
    await browser_session.goto(_data_url(
        "<style>body{margin:0} #b{display:block;width:600px;height:400px;margin:100px;background:#ddd;border:0}"
        "#b:hover{background:#c00}</style><button id=b>Hover me</button>"))
    before = await browser_session.screenshot()
    await browser_session.click(*await _center(browser_session, "#b"))
    await browser_session.settle()
    after = await browser_session.screenshot()
    assert after.getpixel((150, 150)) == before.getpixel((150, 150)) == (221, 221, 221)
    assert not browser_session.screen_changed(before, after)


@pytest.mark.asyncio
async def test_click_is_clamped_into_the_viewport(browser_session, fake_sandbox_url, sid):
    await browser_session.goto(f"{fake_sandbox_url}/settings?sid={sid}")
    await browser_session.click(-50, 5000)
    await browser_session.click(W + 10, -3)
    with pytest.raises(ValueError):
        await browser_session.click(float("nan"), 10)


@pytest.mark.asyncio
async def test_select_all_then_type_replaces_text(browser_session, fake_sandbox_url, sid):
    await browser_session.goto(f"{fake_sandbox_url}/settings?sid={sid}")
    before = await browser_session.screenshot()
    await browser_session.click(*await _center(browser_session, "#name"))
    await browser_session.select_all()
    await browser_session.type("Alex Kim")
    await browser_session.settle()
    assert await browser_session.page.locator("#name").input_value() == "Alex Kim"
    assert browser_session.screen_changed(before, await browser_session.screenshot())

    await browser_session.click(*await _center(browser_session, "button.primary"))
    await browser_session.settle()
    assert _state(fake_sandbox_url, sid)["profile"]["name"] == "Alex Kim"


@pytest.mark.asyncio
async def test_press_and_scroll(browser_session):
    await browser_session.goto(_data_url(
        "<body style='margin:0'><input id=f><div style='height:5000px;"
        "background:linear-gradient(#fff,#036)'></div></body>"))
    page = browser_session.page
    await browser_session.press("Tab")
    await browser_session.press("Escape")
    with pytest.raises(PlaywrightError):
        await browser_session.press("NotAKey")

    await browser_session.scroll("down", "small")
    await browser_session.settle()
    y1 = await page.evaluate("window.scrollY")
    await browser_session.scroll("down", "page")
    await browser_session.settle()
    y2 = await page.evaluate("window.scrollY")
    await browser_session.scroll("up", "small")
    await browser_session.settle()
    y3 = await page.evaluate("window.scrollY")
    assert y1 == pytest.approx(240, abs=2)
    assert y2 - y1 == pytest.approx(0.85 * H, abs=2)
    assert y3 == pytest.approx(y2 - 240, abs=2)

    with pytest.raises(ValueError):
        await browser_session.scroll("left", "small")
    with pytest.raises(ValueError):
        await browser_session.scroll("down", "huge")


@pytest.mark.asyncio
async def test_sessions_are_isolated(browser_pool, fake_sandbox_url):
    a, b = await browser_pool.new_session(Viewport()), await browser_pool.new_session(Viewport())
    try:
        await a.goto(f"{fake_sandbox_url}/settings?sid=iso-a")
        await b.goto(f"{fake_sandbox_url}/settings?sid=iso-b")
        assert {c["name"]: c["value"] for c in await a.context.cookies()}["sid"] == "iso-a"
        assert {c["name"]: c["value"] for c in await b.context.cookies()}["sid"] == "iso-b"

        await a.click(*await _center(a, "#hc"))
        await a.settle()
        assert _state(fake_sandbox_url, "iso-a")["settings"]["accessibility"]["high_contrast"] is True
        assert _state(fake_sandbox_url, "iso-b")["settings"]["accessibility"]["high_contrast"] is False

        await b.goto(f"{fake_sandbox_url}/settings")  # no ?sid: b's cookie still says iso-b
        assert await b.page.locator("#hc.on").count() == 0
    finally:
        await a.close()
        await b.close()


@pytest.mark.asyncio
async def test_js_dialogs_are_accepted(browser_session):
    await browser_session.goto(_data_url(
        "<button id=d style='width:400px;height:200px' "
        "onclick=\"document.title = confirm('Delete it?') ? 'accepted' : 'dismissed'\">Delete</button>"))
    await browser_session.click(*await _center(browser_session, "#d"))
    await browser_session.settle()
    assert await browser_session.title() == "accepted"
    assert browser_session.dialogs == [{"type": "confirm", "message": "Delete it?"}]


@pytest.mark.asyncio
async def test_new_tab_replaces_the_page(browser_session, fake_sandbox_url, sid):
    await browser_session.goto(_data_url(
        f"<a id=l target=_blank href='{fake_sandbox_url}/settings?sid={sid}' "
        "style='display:block;width:500px;height:300px'>Open settings</a>"))
    first = browser_session.page
    await browser_session.click(*await _center(browser_session, "#l"))
    await browser_session.settle()
    assert browser_session.page is not first and first.is_closed()
    assert browser_session.context.pages == [browser_session.page]
    assert await browser_session.title() == "Settings - Acme Utilities"
    assert (await browser_session.screenshot()).size == (W, H)


@pytest.mark.asyncio
async def test_close_is_idempotent(browser_pool):
    session = await browser_pool.new_session(Viewport())
    await session.close()
    await session.close()


@pytest.mark.asyncio
async def test_pool_is_a_per_loop_singleton(browser_cfg):
    p1 = await get_pool(browser_cfg)
    try:
        assert await get_pool(browser_cfg) is p1 and p1.connected
        s1 = await p1.new_session(Viewport())
        # Other per-session settings share the Chromium and never close a live run's context.
        p2 = await get_pool(BrowserConfig(headless=True, settle_ms=100, locale="de-DE"))
        s2 = await p2.new_session(Viewport())
        assert p1.connected and p2.connected
        assert s1.context.browser is s2.context.browser
        assert (s1.cfg.settle_ms, s2.cfg.settle_ms) == (browser_cfg.settle_ms, 100)
        await s2.goto(_data_url("<p>x</p>"))
        assert await s2.page.evaluate("navigator.language") == "de-DE"
        assert (await s1.screenshot()).size == (W, H)  # still alive
        await close_pool()
        await close_pool()
        p3 = await get_pool(browser_cfg)
        assert p3 is not p1 and p3.connected and not p1.connected
    finally:
        await close_pool()


@pytest.mark.asyncio
async def test_sessions_starting_together_after_a_crash_share_one_new_browser(browser_cfg):
    pool = await get_pool(browser_cfg)
    try:
        await pool._browser.close()  # simulate a crash
        assert not pool.connected
        a, b = await asyncio.gather(pool.new_session(Viewport()), pool.new_session(Viewport()))
        assert pool.connected and a.context.browser is b.context.browser is pool._browser
    finally:
        await close_pool()


def test_pool_is_recreated_on_a_new_event_loop(browser_cfg):
    """A live pool from another loop is left alone (it can't be used from here) and replaced."""
    async def second_loop():
        pool = await get_pool(browser_cfg)
        try:
            await (await pool.new_session(Viewport())).close()
            return pool
        finally:
            await close_pool()

    loop1 = asyncio.new_event_loop()
    try:
        p1 = loop1.run_until_complete(get_pool(browser_cfg))
        p2 = asyncio.run(second_loop())
        assert p2 is not p1 and p1.connected
    finally:
        loop1.run_until_complete(p1.close())
        loop1.close()
