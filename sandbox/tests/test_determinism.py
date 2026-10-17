"""Deterministic rendering (docs/tasks/T1-app.md, "Validation"): every page state, shot in
two fresh browser contexts with two fresh sids, gives pixel-identical screenshots, both the
1280x800 viewport and the full page.

Scenes are built with the shared helpers in sandbox/tools/make_gallery.py (the same code
that makes the committed gallery). DOM selectors and AcmeTestHooks are fine here: tests only.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image, ImageChops

from sandbox.tests.conftest import CONTEXT_OPTS, VIEWPORT
from sandbox.tools import make_gallery as mg
from sandbox.tools.make_gallery import Scene

TEXT_L = mg.TEXT_L
HC = mg.HIGH_CONTRAST

SCENES: list[Scene] = [
    Scene("dashboard", "/"),
    Scene("billing", "/billing"),
    Scene("billing-pay-dialog", "/billing?pay=1"),
    Scene("usage", "/usage"),
    Scene("usage-july-selected", "/usage", setup=mg.select_july),
    Scene("outages", "/outages"),
    Scene("outages-riverside-dialog", "/outages", setup=mg.open_riverside),
    Scene("messages-inbox", "/messages"),
    Scene("messages-sent", "/messages?tab=sent"),
    Scene("message-landlord", "/messages/msg-landlord"),
    Scene("compose-recipient-menu", "/messages/compose", setup=mg.open_compose_menu),
    Scene("settings-profile", "/settings?tab=profile"),
    Scene("settings-notifications", "/settings?tab=notifications"),
    Scene("settings-accessibility", "/settings?tab=accessibility"),
    Scene("help-faq1-open", "/help", setup=mg.open_faq_1),
    Scene("signed-out", "/", actions=(("session.sign_out", {}),)),
    Scene("not-found", "/nope"),
    Scene("dashboard-text-L", "/", actions=TEXT_L),
    Scene("billing-text-L", "/billing", actions=TEXT_L),
    Scene("dashboard-high-contrast", "/", actions=HC),
    Scene("settings-notifications-high-contrast", "/settings?tab=notifications", actions=HC),
    Scene("dashboard-nav-collapsed", "/", actions=(("nav.toggle", {}),)),
]

# What each scene must show, so a broken setup can't pass by rendering the wrong page
# twice. (selector that must be visible, or a JS predicate on the page).
EXPECT = {
    "billing-pay-dialog": '[data-testid="pay-dialog"]',
    "usage-july-selected": '[data-testid="usage-details"] .details-month:text-is("July 2026")',
    "outages-riverside-dialog": '[data-testid="report-title"]:text-is("Report an outage in Riverside")',
    "compose-recipient-menu": '[data-testid="compose-to-menu"]',
    "help-faq1-open": '[data-testid="faq-panel-1"]',
    "signed-out": '[data-testid="signin"]',
    "not-found": '#nf-title',
    "message-landlord": '#message-subject:text-is("Water heater replacement on September 20")',
}
EXPECT_JS = {
    "dashboard-text-L": "() => getComputedStyle(document.documentElement).fontSize === '20px'",
    "billing-text-L": "() => getComputedStyle(document.documentElement).fontSize === '20px'",
    "dashboard-high-contrast": "() => document.documentElement.classList.contains('hc')",
    "settings-notifications-high-contrast": "() => document.documentElement.classList.contains('hc')",
    "dashboard-nav-collapsed": "() => !document.querySelector('[data-testid=\"sidenav\"]')",
    "dashboard": "() => !!document.querySelector('[data-testid=\"sidenav\"]') && getComputedStyle(document.documentElement).fontSize === '16px'",
}


def _decode(png: bytes) -> Image.Image:
    return Image.open(io.BytesIO(png)).convert("RGBA")


def _shoot(browser, base_url: str, sid: str, scene: Scene) -> tuple[Image.Image, Image.Image]:
    ctx, page = mg.open_scene(browser, base_url, sid, scene, context_opts=CONTEXT_OPTS)
    try:
        assert page.viewport_size == VIEWPORT
        if scene.name in EXPECT:
            assert page.locator(EXPECT[scene.name]).first.is_visible(), f"{scene.name}: {EXPECT[scene.name]} not visible"
        if scene.name in EXPECT_JS:
            assert page.evaluate(EXPECT_JS[scene.name]), f"{scene.name}: {EXPECT_JS[scene.name]} is false"
        viewport = _decode(page.screenshot())
        full = _decode(page.screenshot(full_page=True))
    finally:
        ctx.close()
    return viewport, full


def _assert_identical(a: Image.Image, b: Image.Image, what: str) -> None:
    assert a.size == b.size, f"{what}: sizes differ {a.size} vs {b.size}"
    bbox = ImageChops.difference(a, b).getbbox()
    assert bbox is None, f"{what}: pixels differ inside {bbox}"


@pytest.mark.parametrize("scene", SCENES, ids=[s.name for s in SCENES])
def test_two_fresh_sessions_pixel_identical(browser, sandbox_url, reset, scene):
    (v1, f1), (v2, f2) = (_shoot(browser, sandbox_url, reset(), scene) for _ in range(2))
    assert v1.size == (VIEWPORT["width"], VIEWPORT["height"])
    assert f1.width == VIEWPORT["width"] and f1.height >= VIEWPORT["height"]
    # Not a blank page: real UI has far more than 16 distinct colors.
    assert v1.getcolors(16) is None, f"{scene.name}: viewport screenshot is nearly blank"
    _assert_identical(v1, v2, f"{scene.name} viewport")
    _assert_identical(f1, f2, f"{scene.name} full page")


def test_scene_ids_unique():
    assert len({s.name for s in SCENES}) == len(SCENES) == 22
    assert set(EXPECT) | set(EXPECT_JS) <= {s.name for s in SCENES}


def test_gallery_committed():
    """The committed gallery (python -m sandbox.tools.make_gallery) is complete: 1280x800 PNGs."""
    for scene in mg.GALLERY:
        path = mg.DEFAULT_OUT / f"{scene.name}.png"
        assert path.is_file(), f"missing {path}; run python -m sandbox.tools.make_gallery"
        with Image.open(path) as img:
            assert img.format == "PNG" and img.size == (VIEWPORT["width"], VIEWPORT["height"]), path
    assert mg.CONTEXT_OPTS == CONTEXT_OPTS, "make_gallery must shoot with the agent's context options"
