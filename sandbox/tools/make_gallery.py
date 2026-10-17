"""Screenshot gallery of the sandbox for the README and the demo video.

    python -m sandbox.tools.make_gallery [--out sandbox/screenshots]

Starts its own sandbox (uvicorn in a thread on a free 127.0.0.1 port). Each scene gets a
fresh sid and a fresh browser context (1280x800, device_scale_factor=1, reduced motion,
fixed locale and timezone), and is saved as an optimized 1280x800 PNG numbered in story
order. Deterministic: the same code gives the same pixels on every run
(sandbox/tests/test_determinism.py checks this for every page state).

The scene helpers below (Scene, open_scene, settle, click_usage_bar, click_district, ...)
are shared with sandbox/tests. They use DOM selectors and window.AcmeTestHooks, which only
tests and tooling may do. The agent never does.
"""

from __future__ import annotations

import argparse
import io
import socket
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx

VIEWPORT = {"width": 1280, "height": 800}
# Same options as sandbox/tests/conftest.py CONTEXT_OPTS (the agent's browser context).
CONTEXT_OPTS = dict(viewport=VIEWPORT, device_scale_factor=1, reduced_motion="reduce",
                    locale="en-US", timezone_id="America/New_York")
# Park the mouse here before a gallery shot: empty top-bar space, so no hover styles show.
NEUTRAL_MOUSE = (640, 30)
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "screenshots"

Action = tuple[str, dict]
Setup = Callable[["object"], None]  # (playwright Page) -> None


@dataclass(frozen=True)
class Scene:
    """One page state: server-side actions first, then a page path, then in-page setup."""

    name: str
    path: str
    actions: tuple[Action, ...] = ()
    setup: Setup | None = None
    note: str = field(default="", compare=False)


# --- scene building blocks -----------------------------------------------------


def with_sid(url: str, sid: str) -> str:
    return f"{url}{'&' if '?' in url else '?'}sid={sid}"


def reset_session(base_url: str, sid: str, task_id: str | None = None) -> str:
    r = httpx.post(f"{base_url}/api/reset", json={"sid": sid, "task_id": task_id}, timeout=5)
    r.raise_for_status()
    return sid


def post_actions(base_url: str, sid: str, actions: tuple[Action, ...] | list[Action]) -> None:
    """Apply UI actions through POST /api/action (the page JS's own endpoint), then render
    one page so the pending toast is consumed and the screenshot shows no toast."""
    if not actions:
        return
    with httpx.Client(base_url=base_url, cookies={"sid": sid}, timeout=5) as client:
        for kind, payload in actions:
            r = client.post("/api/action", json={"type": kind, "payload": payload})
            if r.status_code != 200 or r.json().get("ok") is not True:
                raise RuntimeError(f"action {kind} {payload} failed: {r.status_code} {r.text}")
        client.get("/").raise_for_status()


def settle(page) -> None:
    """Network idle, web fonts ready (none are used, but this proves it), two frames painted."""
    page.wait_for_load_state("networkidle")
    page.evaluate("() => document.fonts.ready.then(() => document.fonts.status)")
    page.evaluate("() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r(true))))")


def _canvas_point(page, canvas_testid: str, hook: str, item_id: str) -> tuple[float, float]:
    canvas = page.locator(f'[data-testid="{canvas_testid}"]')
    canvas.scroll_into_view_if_needed()
    points = page.evaluate(f"() => window.AcmeTestHooks.{hook}")
    pt = next((p for p in points or [] if p["id"] == item_id), None)
    if pt is None:
        raise RuntimeError(f"AcmeTestHooks.{hook} has no {item_id!r}: {points}")
    box = canvas.bounding_box()
    return box["x"] + pt["x"], box["y"] + pt["y"]


def click_usage_bar(page, month: str) -> None:
    """Click a bar on the usage <canvas>; the page reloads with that month selected."""
    x, y = _canvas_point(page, "usage-chart", "usageBars", month)
    with page.expect_navigation():
        page.mouse.click(x, y)
    settle(page)
    page.locator('[data-testid="usage-details"] .details-month').wait_for(state="visible")


def click_district(page, district: str) -> None:
    """Click a district on the outage-map <canvas>; the report dialog opens (no reload)."""
    x, y = _canvas_point(page, "outage-map", "districts", district)
    page.mouse.click(x, y)
    page.locator('[data-testid="report-dialog"]').wait_for(state="visible")


def select_july(page) -> None:
    click_usage_bar(page, "2026-07")


def open_riverside(page) -> None:
    click_district(page, "riverside")


def open_pay_dialog(page) -> None:
    page.click('[data-testid="pay-open"]')
    page.locator('[data-testid="pay-dialog"]').wait_for(state="visible")


def pay_confirm_50(page) -> None:
    """Pay flow up to the confirm dialog for $50.00 (not confirmed)."""
    open_pay_dialog(page)
    page.click('[data-testid="pay-other"]')
    page.fill('[data-testid="pay-amount"]', "50")
    page.click('[data-testid="pay-continue"]')
    page.locator('[data-testid="confirm-dialog"]').wait_for(state="visible")


def open_compose_menu(page) -> None:
    page.click('[data-testid="compose-to"]')
    page.locator('[data-testid="compose-to-menu"]').wait_for(state="visible")


def compose_support(page) -> None:
    """Recipient chosen and text typed, not sent."""
    open_compose_menu(page)
    page.click('[data-testid="compose-to-support"]')
    page.fill('[data-testid="compose-subject"]', "My meter is broken")
    page.fill('[data-testid="compose-body"]', "Hi, my electric meter display is blank and the reading hasn't changed since Tuesday. Can someone take a look?")


def open_faq_1(page) -> None:
    page.click('[data-testid="faq-1"]')
    page.locator('[data-testid="faq-panel-1"]').wait_for(state="visible")


def open_scene(browser, base_url: str, sid: str, scene: Scene, context_opts: dict | None = None,
               park_mouse: bool = False):
    """Fresh context for an already-reset sid, at the scene's state. Returns (context, page);
    the caller closes the context."""
    post_actions(base_url, sid, scene.actions)
    ctx = browser.new_context(**(context_opts or CONTEXT_OPTS))
    try:
        page = ctx.new_page()
        page.goto(with_sid(f"{base_url}{scene.path}", sid))
        settle(page)
        if scene.setup:
            scene.setup(page)
            settle(page)
        if park_mouse:
            page.mouse.move(*NEUTRAL_MOUSE)
            settle(page)
    except BaseException:
        ctx.close()
        raise
    return ctx, page


# --- the gallery -----------------------------------------------------------------

TEXT_L = (("settings.accessibility", {"key": "text_size", "value": "L"}),)
HIGH_CONTRAST = (("settings.accessibility", {"key": "high_contrast", "value": True}),)

GALLERY: list[Scene] = [
    Scene("01-dashboard", "/"),
    Scene("02-billing", "/billing"),
    Scene("03-billing-pay-confirm", "/billing", setup=pay_confirm_50, note="confirm dialog for $50"),
    Scene("04-usage-july", "/usage", setup=select_july, note="July selected via the canvas"),
    Scene("05-outage-report", "/outages", setup=open_riverside, note="Riverside report dialog"),
    Scene("06-messages-inbox", "/messages"),
    Scene("07-compose", "/messages/compose", setup=compose_support, note="recipient chosen, text typed"),
    Scene("08-settings-notifications", "/settings?tab=notifications"),
    Scene("09-settings-accessibility", "/settings?tab=accessibility"),
    Scene("10-dashboard-high-contrast", "/", actions=HIGH_CONTRAST),
    Scene("11-dashboard-text-large", "/", actions=TEXT_L),
    Scene("12-help", "/help", setup=open_faq_1, note="FAQ 1 open"),
]


def png_bytes(screenshot: bytes) -> bytes:
    """Re-encode a Playwright PNG as an optimized RGB PNG (same pixels, smaller file)."""
    from PIL import Image

    img = Image.open(io.BytesIO(screenshot)).convert("RGB")
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_sandbox():
    """uvicorn on a free 127.0.0.1 port in a daemon thread. Returns (base_url, server, thread)."""
    import uvicorn

    from sandbox.app import app

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(200):
        try:
            if httpx.get(f"{url}/api/health", timeout=0.5).status_code == 200:
                return url, server, thread
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    raise RuntimeError("sandbox server did not start")


def make_gallery(out: Path, scenes: list[Scene] = GALLERY) -> list[Path]:
    from playwright.sync_api import sync_playwright

    out.mkdir(parents=True, exist_ok=True)
    url, server, thread = start_sandbox()
    written = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                for i, scene in enumerate(scenes, 1):
                    sid = reset_session(url, f"gallery-{i:02d}")
                    ctx, page = open_scene(browser, url, sid, scene, park_mouse=True)
                    try:
                        data = png_bytes(page.screenshot())
                    finally:
                        ctx.close()
                    path = out / f"{scene.name}.png"
                    path.write_bytes(data)
                    written.append(path)
                    print(f"{path}  {len(data) / 1024:.0f} KB")
            finally:
                browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=5)
    total = sum(p.stat().st_size for p in written)
    print(f"{len(written)} screenshots, {total / 1024:.0f} KB total")
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory (default: sandbox/screenshots)")
    args = ap.parse_args(argv)
    make_gallery(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
