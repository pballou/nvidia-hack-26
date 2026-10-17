"""No egress (docs/tasks/T1-app.md, "Validation"): nothing the sandbox UI does leaves 127.0.0.1.

A context-level route('**/*') records every request and aborts any whose host is not
127.0.0.1 (recorded as a violation). The browsing test visits every page state that
test_determinism.py shoots, opens the menus, dialogs, canvases and dropdowns, and performs
real actions. Static checks scan the CSS, JS and templates for off-host URLs. DOM selectors
are fine here: tests only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from sandbox.tests.conftest import CONTEXT_OPTS, with_sid
from sandbox.tools import make_gallery as mg

LOCAL_HOST = "127.0.0.1"
SANDBOX = Path(__file__).resolve().parents[1]
STATIC_FILES = sorted((SANDBOX / "static").glob("*.css")) + sorted((SANDBOX / "static").glob("*.js"))
TEMPLATES = sorted((SANDBOX / "templates").glob("*.html"))
# Chromium logs this for any 4xx/5xx or aborted subresource; intentional 400s from
# /api/action and the 404 page cause it. Real JS errors look different.
NETWORK_NOISE = "Failed to load resource"


@dataclass
class Recorder:
    routed: list[str] = field(default_factory=list)      # every URL the route saw
    requests: list[str] = field(default_factory=list)    # every request event (cross-check)
    violations: list[str] = field(default_factory=list)  # off-host URLs (aborted)
    page_errors: list[str] = field(default_factory=list)
    console_errors: list[str] = field(default_factory=list)
    offhost_refs: list[str] = field(default_factory=list)
    pages_checked: int = 0


def install_guard(context, rec: Recorder) -> None:
    def handle(route):
        url = route.request.url
        rec.routed.append(url)
        if urlsplit(url).hostname != LOCAL_HOST:
            rec.violations.append(url)
            route.abort("blockedbyclient")
        else:
            route.continue_()

    context.route("**/*", handle)
    context.on("request", lambda r: rec.requests.append(r.url))


def watch_page(page, rec: Recorder) -> None:
    page.on("pageerror", lambda e: rec.page_errors.append(f"{page.url}: {e}"))

    def on_console(msg):
        if msg.type == "error" and NETWORK_NOISE not in msg.text:
            rec.console_errors.append(f"{page.url}: {msg.text}")

    page.on("console", on_console)


@pytest.fixture
def guarded(browser, reset):
    """(page, rec, sid): a fresh context with the egress guard, a fresh sid, error watchers."""
    rec = Recorder()
    ctx = browser.new_context(**CONTEXT_OPTS)
    install_guard(ctx, rec)
    page = ctx.new_page()
    watch_page(page, rec)
    yield page, rec, reset()
    ctx.close()


# Every URL-bearing attribute and CSS url() in the live DOM, resolved by the browser.
DOM_REFS_JS = r"""() => {
  const out = [];
  const add = (where, v) => { if (v) out.push([where, String(v)]); };
  const sel = 'link[href], script[src], img[src], img[srcset], source[src], source[srcset], iframe[src], ' +
    'video[src], video[poster], audio[src], embed[src], object[data], input[type=image][src], track[src], ' +
    'a[href], area[href], form[action], button[formaction], base[href], meta[http-equiv]';
  document.querySelectorAll(sel).forEach((el) => {
    const tag = el.tagName.toLowerCase();
    for (const a of ['href', 'src', 'poster', 'data', 'action', 'formAction']) {
      if (el.hasAttribute(a.toLowerCase())) add(tag + '[' + a + ']', el[a] || el.getAttribute(a.toLowerCase()));
    }
    for (const a of ['srcset', 'content']) if (el.hasAttribute(a)) add(tag + '[' + a + ']', el.getAttribute(a));
  });
  document.querySelectorAll('svg image, svg use, svg feImage').forEach((el) => {
    add('svg ' + el.tagName, el.getAttribute('href') || el.getAttribute('xlink:href'));
  });
  const cssUrls = (text, where) => {
    for (const m of text.matchAll(/url\(\s*(['"]?)([^'")]*)\1\s*\)|@import\s+(['"])([^'"]+)\3/g)) {
      add(where, new URL(m[2] ?? m[4], document.baseURI).href);
    }
  };
  for (const sheet of document.styleSheets) {
    let rules = [];
    try { rules = [...sheet.cssRules]; } catch (e) { add('stylesheet (unreadable)', sheet.href || 'inline'); }
    rules.forEach((r) => cssUrls(r.cssText, 'css ' + (sheet.href || 'inline <style>')));
  }
  document.querySelectorAll('[style]').forEach((el) => cssUrls(el.getAttribute('style'), 'style attr'));
  return out;
}"""


def _offhost(where: str, value: str) -> bool:
    if where == "meta[content]":
        return bool(re.search(r"url\s*=\s*(https?:)?//(?!127\.0\.0\.1[:/])", value, re.I))
    if where.endswith("[srcset]"):
        return any(_offhost("img[src]", part.split()[0]) for part in value.split(",") if part.strip())
    if where == "link[href]" and value == "data:,":
        return False  # the favicon: an empty data URL, not a request
    parts = urlsplit(value)
    if where in ("a[href]", "area[href]") and parts.scheme in ("mailto", "tel"):
        return False
    return parts.scheme not in ("http", "https") or parts.hostname != LOCAL_HOST


def check_dom(page, rec: Recorder, label: str) -> None:
    rec.pages_checked += 1
    for where, value in page.evaluate(DOM_REFS_JS):
        if _offhost(where, value):
            rec.offhost_refs.append(f"{label}: {where} = {value}")


def goto(page, rec: Recorder, base: str, path: str, sid: str | None = None) -> None:
    url = f"{base}{path}"
    page.goto(with_sid(url, sid) if sid else url)
    mg.settle(page)
    check_dom(page, rec, path)


def act(page, rec: Recorder, selector: str, label: str, toast: bool = True) -> None:
    """Click a [data-act] control: Acme.act POSTs, then the page reloads (or redirects) and
    shows the toast (nav.toggle and sign-out have none)."""
    with page.expect_navigation():
        page.click(selector)
    mg.settle(page)
    check_dom(page, rec, label)
    assert page.locator('[data-testid="toast"]').is_visible() == toast, f"{label}: toast visible != {toast}"


def open_menu(page, button: str, menu: str) -> None:
    page.click(f'[data-testid="{button}"]')
    page.locator(f'[data-testid="{menu}"]').wait_for(state="visible")
    page.keyboard.press("Escape")
    page.locator(f'[data-testid="{menu}"]').wait_for(state="hidden")


def test_browsing_everything_stays_on_localhost(guarded, sandbox_url, get_state):
    page, rec, sid = guarded
    base = sandbox_url

    # Dashboard, top-bar menus.
    goto(page, rec, base, "/", sid)
    open_menu(page, "topbar-bell", "bell-menu")
    open_menu(page, "topbar-avatar", "avatar-menu")

    # Billing: pay dialog (2 steps, confirmed), autopay, download.
    goto(page, rec, base, "/billing")
    mg.pay_confirm_50(page)
    check_dom(page, rec, "/billing confirm dialog")
    act(page, rec, '[data-testid="pay-confirm"]', "billing.pay")
    act(page, rec, '[data-testid="autopay-toggle"]', "billing.autopay")
    act(page, rec, '[data-testid="download-2026-09"]', "billing.download")
    goto(page, rec, base, "/billing?pay=1")
    page.locator('[data-testid="pay-dialog"]').wait_for(state="visible")
    page.keyboard.press("Escape")

    # Usage: click a canvas bar (reloads with the month selected).
    goto(page, rec, base, "/usage")
    mg.select_july(page)
    check_dom(page, rec, "/usage july")

    # Outages: click a district (dialog), cancel; report another one.
    goto(page, rec, base, "/outages")
    mg.open_riverside(page)
    check_dom(page, rec, "/outages riverside dialog")
    page.click('[data-testid="report-cancel"]')
    mg.click_district(page, "hillcrest")
    act(page, rec, '[data-testid="report-submit"]', "outages.report")

    # Messages: star, sent tab, open a message, compose with the dropdown, send.
    goto(page, rec, base, "/messages")
    act(page, rec, '[data-testid="star-msg-landlord"]', "messages.star")
    goto(page, rec, base, "/messages?tab=sent")
    goto(page, rec, base, "/messages/msg-landlord")
    goto(page, rec, base, "/messages/compose")
    mg.compose_support(page)
    check_dom(page, rec, "/messages/compose filled")
    act(page, rec, '[data-testid="compose-send"]', "messages.send")
    assert urlsplit(page.url).path == "/messages" and "tab=sent" in page.url

    # Settings: every tab, a notification switch, text size, high contrast.
    goto(page, rec, base, "/settings?tab=profile")
    goto(page, rec, base, "/settings?tab=notifications")
    act(page, rec, '[data-testid="notif-billing-sms"]', "settings.notification")
    goto(page, rec, base, "/settings?tab=accessibility")
    act(page, rec, '[data-testid="a11y-text-size-L"]', "settings.accessibility text_size")

    # Help: an FAQ, and the call-back form.
    goto(page, rec, base, "/help")
    mg.open_faq_1(page)
    page.click('[data-testid="callback-topic-billing"]')
    page.fill('[data-testid="callback-message"]', "Please call me about my September bill.")
    act(page, rec, '[data-testid="callback-submit"]', "help.contact")

    # Unknown page (the app's own 404 page).
    goto(page, rec, base, "/nope")

    # Text size L pages, then high contrast, then the collapsed nav.
    goto(page, rec, base, "/")
    goto(page, rec, base, "/billing")
    goto(page, rec, base, "/settings?tab=accessibility")
    act(page, rec, '[data-testid="a11y-high-contrast"]', "settings.accessibility high_contrast")
    goto(page, rec, base, "/")
    goto(page, rec, base, "/settings?tab=notifications")
    act(page, rec, '[data-testid="topbar-menu"]', "nav.toggle", toast=False)
    assert page.locator('[data-testid="sidenav"]').count() == 0

    # Sign out from the avatar menu, then the signed-out page's Sign in.
    page.click('[data-testid="topbar-avatar"]')
    act(page, rec, '[data-testid="menu-signout"]', "session.sign_out", toast=False)
    assert page.locator('[data-testid="signin"]').is_visible()
    act(page, rec, '[data-testid="signin"]', "session.sign_in")

    state = get_state(sid)
    done = {h["type"] for h in state["history"]}
    required = {"billing.autopay", "billing.download", "messages.star", "settings.accessibility", "settings.notification"}
    assert required <= done, f"missing actions: {required - done}"
    assert state["meta"]["action_count"] >= 13

    assert rec.violations == [], f"off-host requests: {rec.violations}"
    hosts = {urlsplit(u).hostname for u in rec.routed + rec.requests}
    assert hosts == {LOCAL_HOST}, hosts
    assert set(rec.requests) <= set(rec.routed), "a request bypassed the route"
    assert len(rec.routed) >= 80 and rec.pages_checked >= 30, (len(rec.routed), rec.pages_checked)
    assert rec.offhost_refs == [], rec.offhost_refs
    assert rec.page_errors == [], rec.page_errors
    assert rec.console_errors == [], rec.console_errors
    print(f"\nno-egress: 0 violations over {len(rec.routed)} requests, {rec.pages_checked} DOM checks")


def test_guard_catches_egress(guarded, sandbox_url):
    """The guard is not vacuous: off-host fetches and images are recorded and aborted."""
    page, rec, sid = guarded
    goto(page, rec, sandbox_url, "/", sid)
    result = page.evaluate("() => fetch('http://example.invalid/x').then(() => 'loaded', () => 'blocked')")
    page.evaluate("""() => new Promise((done) => {
      const img = new Image();
      img.onload = img.onerror = () => done(true);
      img.src = 'https://cdn.example.invalid/pixel.png';
      document.body.append(img);
    })""")
    assert result == "blocked"
    assert rec.violations == ["http://example.invalid/x", "https://cdn.example.invalid/pixel.png"]
    check_dom(page, rec, "/ with an injected image")
    assert rec.offhost_refs == ["/ with an injected image: img[src] = https://cdn.example.invalid/pixel.png"]
    assert rec.page_errors == [] and rec.console_errors == []


# --- static scans --------------------------------------------------------------

ABS_URL = re.compile(r"https?://[^\s'\"<>)]*", re.I)
# XML namespace names are identifiers, never fetched.
NAMESPACES = {"http://www.w3.org/2000/svg", "http://www.w3.org/1999/xlink"}
CSS_URL = re.compile(r"url\(\s*(['\"]?)([^'\")]*)\1\s*\)|@import\s+(['\"])([^'\"]+)\3", re.I)
PROTOCOL_RELATIVE_ATTR = re.compile(r"\b(?:src|href|action|poster|data|srcset|formaction)\s*=\s*['\"]?\s*//", re.I)
JS_PROTOCOL_RELATIVE = re.compile(r"['\"`]//")
REMOTE_APIS = re.compile(r"\b(?:WebSocket|EventSource|RTCPeerConnection|importScripts|sendBeacon)\b|\bimport\s*\(")


def _rel(p: Path) -> str:
    return str(p.relative_to(SANDBOX))


def test_static_files_exist():
    assert any(p.suffix == ".css" for p in STATIC_FILES) and any(p.suffix == ".js" for p in STATIC_FILES)
    assert len(TEMPLATES) >= 10


@pytest.mark.parametrize("path", STATIC_FILES + TEMPLATES, ids=_rel)
def test_no_absolute_urls(path):
    text = path.read_text()
    found = [u for u in ABS_URL.findall(text) if u.rstrip("/\"'") not in NAMESPACES]
    assert found == [], f"{_rel(path)} has absolute URLs: {found}"
    assert not PROTOCOL_RELATIVE_ATTR.search(text), f"{_rel(path)} has a protocol-relative src/href"


@pytest.mark.parametrize("path", [p for p in STATIC_FILES if p.suffix == ".css"] + TEMPLATES, ids=_rel)
def test_css_urls_local(path):
    """Every CSS url()/@import (in .css files and in templates' <style>/style=) is local."""
    bad = []
    for m in CSS_URL.finditer(path.read_text()):
        target = (m.group(2) if m.group(2) is not None else m.group(4)).strip()
        if target.startswith("data:") or target.startswith("#"):
            continue
        if target.startswith("//") or urlsplit(target).scheme or not target:
            bad.append(target)
    assert bad == [], f"{_rel(path)}: non-local CSS urls {bad}"


@pytest.mark.parametrize("path", [p for p in STATIC_FILES if p.suffix == ".js"], ids=_rel)
def test_js_has_no_remote_endpoints(path):
    text = path.read_text()
    assert not JS_PROTOCOL_RELATIVE.search(text), f"{_rel(path)}: protocol-relative URL string"
    assert not REMOTE_APIS.search(text), f"{_rel(path)}: uses a remote-capable API"
    for m in re.finditer(r"fetch\(\s*(['\"`])([^'\"`]*)\1", text):
        assert m.group(2).startswith("/") and not m.group(2).startswith("//"), f"{_rel(path)}: fetch({m.group(2)!r})"
