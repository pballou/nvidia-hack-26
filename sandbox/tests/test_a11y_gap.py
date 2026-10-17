"""The accessibility gap is real, and the layout rules hold (sandbox/README.md "UI rules").

- The deliberately unlabeled icon controls have no accessible name: no aria-label,
  aria-labelledby or title, no text, no SVG <title>, and both Playwright's aria snapshot
  and Chromium's own accessible-name computation (CDP) give them an empty name. The
  labeled top-bar controls do have names. In the accessibility tree the 12 download
  buttons are identical nameless buttons, and the 14 stars too (row context aside).
- Canvas widgets have no DOM element per bar or district.
- Fold rules at 1280x800, text size M: what must be visible without scrolling, and what
  must need a scroll (the scroll tasks).
- Nothing renders outside the page's pixels: no native <select>, date/color pickers,
  title tooltips or alert/confirm/prompt. The static JS has no clocks, randomness or timers.
- Text size and high contrast really restyle the page.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from sandbox import state as st

STATIC = Path(__file__).resolve().parents[1] / "static"
VIEW_W, VIEW_H = 1280, 800

STATEMENT_IDS = [s["id"] for s in st.default_state()["billing"]["statements"]]
MESSAGE_IDS = list(st.default_state()["messages"]["inbox"])
MONTHS = st.default_state()["usage"]["months"]
DISTRICT_NAMES = [name for _, name in st.DISTRICTS]

UNLABELED = {
    "/": ["topbar-menu", "topbar-settings"],
    "/billing": [f"download-{sid}" for sid in STATEMENT_IDS],
    "/messages": ["mark-all-read"] + [f"star-{mid}" for mid in MESSAGE_IDS],
}
LABELED = {"topbar-bell": "Notifications", "topbar-help": "Help", "topbar-avatar": "Account menu"}

# Attributes, text and SVG children that could name a control, read from the DOM.
NAME_SOURCES_JS = """(el) => ({
  tag: el.tagName.toLowerCase(),
  attrs: ['aria-label', 'aria-labelledby', 'aria-describedby', 'aria-description', 'title', 'alt']
    .filter((a) => el.hasAttribute(a)),
  text: el.textContent.replace(/\\s+/g, ''),
  labels: el.labels ? el.labels.length : 0,
  svgNamers: el.querySelectorAll('title, desc, [aria-label], [aria-labelledby], [title], img, [alt]').length,
  svgs: [...el.querySelectorAll('svg')].map((s) => s.getAttribute('aria-hidden')),
})"""


def ax_name(page, testid: str) -> str:
    """Chromium's computed accessible name for the element (CDP Accessibility domain)."""
    cdp = page.context.new_cdp_session(page)
    try:
        root = cdp.send("DOM.getDocument", {"depth": 0})["root"]["nodeId"]
        node = cdp.send("DOM.querySelector", {"nodeId": root, "selector": f'[data-testid="{testid}"]'})["nodeId"]
        assert node, f"{testid} not found"
        ax = cdp.send("Accessibility.getPartialAXTree", {"nodeId": node, "fetchRelatives": False})["nodes"][0]
        return (ax.get("name") or {}).get("value", "")
    finally:
        cdp.detach()


def snapshot_head(page, testid: str) -> str:
    return page.get_by_test_id(testid).aria_snapshot().splitlines()[0]


# ---- the accessibility gap -------------------------------------------------------------


@pytest.mark.parametrize("path", list(UNLABELED))
def test_icon_controls_have_no_accessible_name(path, open_page):
    page, _ = open_page(path=path)
    for testid in UNLABELED[path]:
        loc = page.get_by_test_id(testid)
        assert loc.count() == 1 and loc.is_visible(), f"{path}: {testid} missing or hidden"
        src = loc.evaluate(NAME_SOURCES_JS)
        assert src["tag"] in ("button", "a"), f"{testid} is a <{src['tag']}>"
        assert src["attrs"] == [], f"{testid} has naming attributes {src['attrs']}"
        assert src["text"] == "", f"{testid} has text {src['text']!r}"
        assert src["labels"] == 0, f"{testid} has a <label>"
        assert src["svgNamers"] == 0, f"{testid} has an SVG <title>/<desc> or a named child"
        assert src["svgs"] and all(a == "true" for a in src["svgs"]), f"{testid}: icon SVG not aria-hidden"
        # Playwright's a11y tree: role only, no quoted name ('- button', '- button [expanded]', '- link:').
        head = snapshot_head(page, testid)
        assert re.fullmatch(r'- (button|link)( \[[^\]"]+\])*:?', head), f"{testid} has a name in the a11y tree: {head!r}"
        assert ax_name(page, testid) == "", f"{testid} has computed accessible name {ax_name(page, testid)!r}"


def test_labeled_topbar_controls_have_names(open_page):
    page, _ = open_page(path="/")
    for testid, name in LABELED.items():
        head = snapshot_head(page, testid)
        assert re.match(rf'- (button|link) "{re.escape(name)}"', head), f"{testid}: {head!r}"
        assert ax_name(page, testid) == name


def test_unlabeled_controls_are_indistinguishable_in_a11y_tree(open_page):
    """A DOM agent sees 12 identical download buttons and 14 identical stars."""
    page, _ = open_page(path="/billing")
    table = page.get_by_test_id("statements-table").aria_snapshot()
    nameless = [line for line in table.splitlines() if line.strip() == "- button"]
    assert len(nameless) == len(STATEMENT_IDS), table
    page, _ = open_page(path="/messages")
    names = {page.get_by_test_id(f"star-{mid}").aria_snapshot() for mid in MESSAGE_IDS if mid != "msg-support-reply"}
    assert names == {"- button"}, names  # msg-support-reply starts starred: '- button [pressed]'


# ---- canvases ------------------------------------------------------------------------------

CANVAS_DOM_JS = """([canvasId, containerSel]) => {
  const c = document.querySelector(`[data-testid="${canvasId}"]`);
  const box = c.closest(containerSel);
  const exact = (root) => [...root.querySelectorAll('*')].filter((e) => e.tagName !== 'SCRIPT')
    .map((e) => e.textContent.trim());
  return {
    tag: c.tagName.toLowerCase(),
    children: c.childElementCount,
    inner: c.innerHTML.trim(),
    outer: c.outerHTML,
    parentKids: c.parentElement.childElementCount,
    containerText: box.innerText,
    containerTexts: exact(box),
    pageTexts: exact(document.body),
    // Anything a DOM agent could click or read as a target, and any per-item data attribute.
    interactive: [...document.querySelectorAll('a, button, input, textarea, [role], [tabindex]:not([tabindex="-1"])')]
      .map((e) => (e.getAttribute('aria-label') || '') + ' ' + e.textContent.trim()),
    itemAttrs: [...document.querySelectorAll('body *')].flatMap((e) => [...e.attributes]
      .filter((a) => /month|district|bar|kwh/i.test(a.name)).map((a) => a.name)),
  };
}"""


def test_usage_chart_is_one_canvas(open_page):
    page, _ = open_page(path="/usage")
    dom = page.evaluate(CANVAS_DOM_JS, ["usage-chart", ".usage-chart-card"])
    assert dom["tag"] == "canvas" and dom["children"] == 0 and dom["inner"] == ""
    assert dom["parentKids"] == 1, "the chart body holds only the canvas"
    labels = {m["short"] for m in MONTHS} | {m["label"] for m in MONTHS}
    assert not labels & set(dom["containerTexts"]), "an element per month exists in the chart card"
    assert not labels & set(dom["pageTexts"]), "an element per month exists on the page"
    assert "Jul" not in dom["containerText"] and "Jul" not in dom["outer"]
    assert dom["itemAttrs"] == [], dom["itemAttrs"]
    for text in dom["interactive"]:  # nothing to click that names the answer month
        assert not re.search(r"\bJul(y)?\b", text), f"an interactive element names July: {text!r}"


def test_outage_map_is_one_canvas(open_page):
    page, _ = open_page(path="/outages")
    dom = page.evaluate(CANVAS_DOM_JS, ["outage-map", ".map-card"])
    assert dom["tag"] == "canvas" and dom["children"] == 0 and dom["inner"] == ""
    assert dom["parentKids"] == 1, "the map card holds only the canvas"
    for name in DISTRICT_NAMES:
        assert name not in dom["containerText"] and name not in dom["outer"], name
        assert not any(name in t for t in dom["containerTexts"]), name
    assert "Riverside" not in dom["pageTexts"]
    assert dom["itemAttrs"] == [], dom["itemAttrs"]
    for text in dom["interactive"]:  # nothing to click that names the target district
        assert "Riverside" not in text, f"an interactive element names Riverside: {text!r}"


@pytest.mark.parametrize("path,testid", [("/usage", "usage-chart"), ("/outages", "outage-map")])
def test_canvases_have_no_keyboard_path(path, testid, open_page, get_state):
    """Mouse-only on purpose: arrow keys + Enter must not pick a bar or a district blind."""
    page, sid = open_page(path=path)
    canvas = page.get_by_test_id(testid)
    assert canvas.get_attribute("tabindex") is None
    canvas.evaluate("el => { el.tabIndex = 0; el.focus(); }")  # even forced into focus, keys do nothing
    for key in ("ArrowRight", "ArrowDown", "Enter", " "):
        page.keyboard.press(key)
    assert page.locator(".modal-backdrop:visible").count() == 0
    state = get_state(sid)
    assert state["usage"]["selected_month"] is None and state["history"] == []


# ---- fold rules -----------------------------------------------------------------------------

FOLD = [
    # (path, fully visible at scrollY 0, entirely below the fold)
    ("/billing", ["statement-row-2026-09", "download-2026-09", "autopay-toggle"], []),
    ("/settings?tab=notifications", ["notif-billing-sms"],
     ["notif-promotions-email", "notif-promotions-sms", "notif-promotions-push"]),
    ("/messages", ["mark-all-read"], ["star-msg-landlord", "msg-row-msg-landlord"]),
    ("/usage", ["usage-chart"], []),
    ("/outages", ["outage-map"], []),
]

GEOM_JS = """(el) => {
  const r = el.getBoundingClientRect();
  const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
  const hit = cx >= 0 && cy >= 0 && cx < innerWidth && cy < innerHeight ? document.elementFromPoint(cx, cy) : null;
  return {left: r.left, top: r.top, right: r.right, bottom: r.bottom, w: r.width, h: r.height,
          unobstructed: !!hit && (el === hit || el.contains(hit))};
}"""


@pytest.mark.parametrize("path,visible,below", FOLD, ids=[f[0] for f in FOLD])
def test_fold_rules(path, visible, below, open_page):
    page, _ = open_page(path=path)
    assert page.viewport_size == {"width": VIEW_W, "height": VIEW_H}
    assert page.evaluate("window.scrollY") == 0
    assert page.evaluate("getComputedStyle(document.documentElement).fontSize") == "16px"
    for testid in visible:
        g = page.get_by_test_id(testid).evaluate(GEOM_JS)
        assert g["w"] > 0 and g["h"] > 0, testid
        assert g["left"] >= 0 and g["top"] >= 0 and g["right"] <= VIEW_W and g["bottom"] <= VIEW_H, f"{testid} not fully visible: {g}"
        assert g["unobstructed"], f"{testid} is covered by another element"
    for testid in below:
        g = page.get_by_test_id(testid).evaluate(GEOM_JS)
        assert g["h"] > 0 and g["top"] >= VIEW_H, f"{testid} should be entirely below the fold: {g}"


# ---- nothing outside the page's pixels ------------------------------------------------------

PAGES = [
    "/", "/billing", "/billing?pay=1", "/usage", "/outages", "/messages", "/messages?tab=sent",
    "/messages/msg-landlord", "/messages/compose", "/settings", "/settings?tab=profile",
    "/settings?tab=notifications", "/settings?tab=accessibility", "/help",
]
BANNED_ELEMENTS = ("select, datalist, input[type=date], input[type=time], input[type=datetime-local], "
                   "input[type=month], input[type=week], input[type=color], input[type=file], "
                   "[title], svg title, a[target]")
FORBIDDEN_JS = re.compile(
    r"\b(alert|confirm|prompt)\s*\(|\bDate\b|Math\.random|performance\.now|\bsetTimeout\b|\bsetInterval\b"
    r"|\brequestAnimationFrame\b|\brequestIdleCallback\b|window\.open\b|\.animate\s*\("
)

POPUP_SCAN_JS = """(banned) => ({
  banned: [...document.querySelectorAll(banned)].map((e) => e.outerHTML.slice(0, 120)),
  handlers: [...document.querySelectorAll('*')].flatMap((e) => [...e.attributes]
    .filter((a) => a.name.startsWith('on') || /^\\s*javascript:/i.test(a.value))
    .map((a) => `<${e.tagName.toLowerCase()} ${a.name}="${a.value}">`)),
  inline: [...document.scripts].filter((s) => !s.src && s.type !== 'application/json').map((s) => s.textContent),
  scripts: [...document.scripts].filter((s) => s.src).map((s) => new URL(s.src).pathname),
})"""


@pytest.mark.parametrize("path", PAGES)
def test_no_native_popups(path, open_page):
    page, _ = open_page(path=path)
    dialogs = []
    page.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))
    assert not page.title().startswith("Not found"), f"{path} rendered the 404 page"
    if path == "/billing?pay=1":
        assert page.get_by_test_id("pay-dialog").is_visible(), "?pay=1 opens the in-page pay dialog"
    scan = page.evaluate(POPUP_SCAN_JS, BANNED_ELEMENTS)
    assert scan["banned"] == [], f"{path}: native widgets or tooltips: {scan['banned']}"
    assert scan["handlers"] == [], f"{path}: inline handlers: {scan['handlers']}"
    for code in scan["inline"]:
        assert not FORBIDDEN_JS.search(strip_js_comments(code)), f"{path}: inline script uses {FORBIDDEN_JS.search(code)}"
    assert all(s.startswith("/static/") and s.endswith(".js") for s in scan["scripts"]), scan["scripts"]
    assert not page.page_errors(filter="all"), page.page_errors(filter="all")
    assert not dialogs, dialogs


# ---- static JS: no popups, clocks, randomness or timers -------------------------------------

_REGEX_AFTER = set("(,=:[!&|?{};+-*%<>~^")
_REGEX_AFTER_WORDS = ("return", "typeof", "case", "do", "else", "in", "of", "void", "delete", "throw")


def strip_js_comments(src: str) -> str:
    """Remove // and /* */ comments, leaving strings, template literals and regex literals intact."""
    out: list[str] = []
    prev = ""  # last significant code character
    i, n = 0, len(src)
    while i < n:
        c, nxt = src[i], src[i + 1] if i + 1 < n else ""
        if c in "'\"`":
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            out.append(src[i:j + 1])
            prev, i = c, j + 1
        elif c == "/" and nxt == "*":
            j = src.find("*/", i + 2)
            out.append(" ")
            i = n if j < 0 else j + 2
        elif c == "/" and nxt == "/":
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif c == "/" and (prev == "" or prev in _REGEX_AFTER or
                           re.search(r"\b(" + "|".join(_REGEX_AFTER_WORDS) + r")\s*$", "".join(out[-12:]))):
            j, in_class = i + 1, False
            while j < n and src[j] != "\n":
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == "[":
                    in_class = True
                elif src[j] == "]":
                    in_class = False
                elif src[j] == "/" and not in_class:
                    break
                j += 1
            out.append(src[i:j + 1])
            prev, i = "/", j + 1
        else:
            out.append(c)
            if not c.isspace():
                prev = c
            i += 1
    return "".join(out)


def test_strip_js_comments():
    src = ("/* no Date here */ const a = 'http://x/*y*/'; // setTimeout(f)\n"
           "const r = /\\/\\/[/]/g; const t = `//${a}`; x = y / 2; // Math.random\nreturn /a/.test(s);")
    out = strip_js_comments(src)
    assert "Date" not in out and "setTimeout" not in out and "Math.random" not in out
    assert "'http://x/*y*/'" in out and "/\\/\\/[/]/g" in out and "`//${a}`" in out
    assert "y / 2" in out and "return /a/.test(s);" in out


JS_FILES = sorted(STATIC.glob("*.js"))


def test_static_js_present():
    assert {p.name for p in JS_FILES} >= {"app.js", "billing.js", "usage.js", "outages.js", "messages.js",
                                          "settings.js", "help.js"}


@pytest.mark.parametrize("path", JS_FILES, ids=[p.name for p in JS_FILES])
def test_static_js_is_deterministic_and_popup_free(path):
    src = path.read_text()
    code = strip_js_comments(src)
    assert len(code) > 0.3 * len(src.strip()), "comment stripping removed too much"
    bad = [(m.group(0), code[:m.start()].count("\n") + 1) for m in FORBIDDEN_JS.finditer(code)]
    assert not bad, f"{path.name}: forbidden calls as (token, line): {bad}"


# ---- text size and contrast are real -------------------------------------------------------


def test_text_size_and_high_contrast_restyle_the_page(open_page):
    page, _ = open_page(path="/settings?tab=accessibility")
    css = lambda js: page.evaluate(js)
    root_px = "getComputedStyle(document.documentElement).fontSize"
    body_bg = "getComputedStyle(document.body).backgroundColor"
    h1_px = "getComputedStyle(document.querySelector('h1')).fontSize"
    assert css(root_px) == "16px" and css(h1_px) == "28px"
    default_bg = css(body_bg)
    assert not css("document.documentElement.classList.contains('hc')")

    for size, px, h1 in (("S", "14px", "24.5px"), ("L", "20px", "35px")):
        with page.expect_navigation():
            page.get_by_test_id(f"a11y-text-size-{size}").click()
        assert css(root_px) == px, f"text size {size}"
        assert css(h1_px) == h1, f"headings rescale with text size {size}"
        assert page.get_by_test_id(f"a11y-text-size-{size}").get_attribute("aria-pressed") == "true"

    with page.expect_navigation():
        page.get_by_test_id("a11y-high-contrast").click()
    assert css("document.documentElement.classList.contains('hc')")
    assert css(body_bg) != default_bg, "high contrast must change the theme"
    assert css(body_bg) == "rgb(0, 0, 0)"
    assert css(root_px) == "20px", "text size survives the contrast change"

    with page.expect_navigation():
        page.get_by_test_id("topbar-logo").click()  # the theme applies on every page
    assert css("document.documentElement.classList.contains('hc')") and css(root_px) == "20px"
