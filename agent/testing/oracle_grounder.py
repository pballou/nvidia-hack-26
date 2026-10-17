"""TEST-ONLY DOM oracle that stands in for the grounder and summary VLM.

!!! NEVER use this in Gym evals, agent metrics or the demo. !!!
The real agent is vision-only (screenshots in, pixel clicks out). This oracle reads the
live page DOM (element boxes and text) so the planner loop can be exercised in CI and on
a laptop without a GPU. Numbers produced with it say nothing about the grounder.

`assert_oracle_allowed()` refuses to run unless under pytest or after
`enable_debug_oracle()` (called only by the CLIs' --use-oracle-grounder debug flag).

`OracleGrounder(browser_session)` implements both Grounder.locate and
Summarizer.summarize (agent/session.py) from `browser_session.page`, a Playwright page.
The screenshot argument is ignored: the page is read directly.

locate: candidate elements (buttons, links, inputs, [role], [onclick], ...) that are
visible and not covered at their click point, scored against the target description:
token matches on the element's own text and labels weigh most, then id/class/data-*/
icon hints, then the text of the surrounding row or section, plus bonuses for type words
("toggle", "icon", ...) and position words ("top-right", ...). Ties prefer the smaller
element. The best match's click point is returned; canvas content can't be resolved.

summarize: the SUMMARY_PROMPT sections of common/grounding.py (PAGE, ELEMENTS, STATE,
OVERLAYS) built from the DOM, so Lightning sees the same shape it gets from the VLM.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from collections import Counter
from typing import Any

from PIL import Image

from agent.session import Located, Summary
from common.grounding import to_grid

MODEL = "oracle"
MAX_LINES = 40

_debug_enabled = False


def enable_debug_oracle() -> None:
    """Allow the oracle outside pytest. Only the --use-oracle-grounder debug flag calls this."""
    global _debug_enabled
    _debug_enabled = True


def assert_oracle_allowed() -> None:
    if "PYTEST_CURRENT_TEST" in os.environ or _debug_enabled:
        return
    raise RuntimeError(
        "the DOM oracle grounder is test-only: it runs only under pytest or with the "
        "--use-oracle-grounder debug flag, and never in Gym evals or the demo"
    )


# --- DOM snapshot (one page.evaluate per call) ---------------------------------------------

_SNAPSHOT_JS = r"""
() => {
  const vw = window.innerWidth, vh = window.innerHeight;
  const norm = s => (s || '').replace(/\s+/g, ' ').trim();
  const clip = (s, n) => { s = norm(s); return s.length > n ? s.slice(0, n - 3).trimEnd() + '...' : s; };
  const textOf = el => norm(el instanceof HTMLElement ? el.innerText : el.textContent);
  const attr = (el, n) => norm(el.getAttribute(n));
  const SEL = 'button, a, input, textarea, select, label, summary, [role], [onclick], [tabindex], ' +
              '[contenteditable], canvas';
  const NATIVE = new Set(['button', 'a', 'input', 'textarea', 'select', 'label', 'summary', 'canvas']);
  const ROLES = new Set(['button', 'link', 'checkbox', 'switch', 'tab', 'menuitem', 'menuitemcheckbox',
    'menuitemradio', 'option', 'radio', 'slider', 'spinbutton', 'textbox', 'combobox', 'searchbox',
    'treeitem', 'gridcell', 'img']);
  const ROW = 'label, li, tr, [role=row], [role=listitem], [role=option], .row, [class*="row"], [class*="item"], ' +
              'fieldset, section, article, dialog, [role=dialog]';
  const LANDMARK = 'header, nav, footer, aside, [role=toolbar], [role=banner], [role=navigation], [role=menubar], ' +
                   '[role=tablist], [role=dialog], dialog';
  const BAR = 'header, nav, footer, aside, [role=banner], [role=navigation], [role=contentinfo]';
  const OVERLAY = '[role=dialog], [role=alertdialog], dialog[open], [role=menu], .modal, .toast, #toast, .popover, ' +
                  '.dropdown-menu, [class*="toast"], [class*="snackbar"]';
  const visible = el => !el.checkVisibility || el.checkVisibility(
    {checkOpacity: true, checkVisibilityCSS: true, opacityProperty: true, visibilityProperty: true});
  const box = el => {
    const r = el.getBoundingClientRect();
    const x0 = Math.max(0, r.left), y0 = Math.max(0, r.top), x1 = Math.min(vw, r.right), y1 = Math.min(vh, r.bottom);
    return (x1 - x0 < 2 || y1 - y0 < 2) ? null : {x0, y0, x1, y1, w: r.width, h: r.height};
  };
  // First point of the visible box where a click would actually reach el (not an overlay).
  const hitPoint = (el, b) => {
    for (const [fx, fy] of [[.5, .5], [.3, .5], [.7, .5], [.5, .3], [.5, .7], [.15, .5], [.85, .5]]) {
      const x = b.x0 + (b.x1 - b.x0) * fx, y = b.y0 + (b.y1 - b.y0) * fy;
      const hit = document.elementFromPoint(x, y);
      if (hit && (hit === el || el.contains(hit))) return [x, y];
    }
    return null;
  };
  const interactive = el => {
    if (NATIVE.has(el.tagName.toLowerCase()) || el.hasAttribute('onclick') || el.isContentEditable) return true;
    const ti = el.getAttribute('tabindex');
    if (ti !== null && Number(ti) >= 0) return true;
    return ROLES.has(attr(el, 'role').toLowerCase());
  };
  const stripOwn = (t, own) => own ? norm(t.replace(own, ' ')) : t;
  const contextOf = (el, own) => {
    const row = el.parentElement && el.parentElement.closest(ROW);
    let t = row && row !== document.body ? stripOwn(textOf(row), own) : '';
    for (let p = el.parentElement, i = 0; !t && p && p !== document.body && i < 3; p = p.parentElement, i++)
      t = stripOwn(textOf(p), own);
    return clip(t, 80);
  };

  const cands = [];
  for (const el of document.querySelectorAll(SEL)) {
    if (cands.length >= 400) break;
    try {
      if (el === document.body || !interactive(el) || !visible(el)) continue;
      const b = box(el); if (!b) continue;
      const pt = hitPoint(el, b); if (!pt) continue;
      cands.push({el, b, pt});
    } catch (e) { /* skip odd elements */ }
  }

  const overlays = [];
  for (const el of document.body.querySelectorAll('*')) {
    try {
      const explicit = el.matches(OVERLAY);
      if (!explicit) {
        const pos = getComputedStyle(el).position;
        if (pos !== 'fixed' && pos !== 'absolute') continue;
      }
      if (!visible(el)) continue;
      const b = box(el); if (!b) continue;
      const t = textOf(el); if (!t) continue;
      const area = (b.x1 - b.x0) * (b.y1 - b.y0);
      if (!explicit && (area < 4000 || area > 0.6 * vw * vh || el.closest(BAR))) continue;
      if (overlays.some(o => o.el.contains(el))) continue;
      const kind = el.matches('[role=dialog], [role=alertdialog], dialog, .modal, [class*="modal"]') ? 'dialog'
        : el.matches('.toast, #toast, [class*="toast"], [class*="snackbar"], [role=status], [role=alert]') ? 'toast'
        : el.matches('[role=menu], [role=listbox], .dropdown-menu, [class*="dropdown"]') ? 'menu' : 'popup';
      overlays.push({el, kind, text: clip(t, 100), cx: (b.x0 + b.x1) / 2, cy: (b.y0 + b.y1) / 2});
    } catch (e) { /* skip */ }
  }

  const els = cands.map(c => c.el);
  const elements = cands.map(({el, b, pt}) => {
    const tag = el.tagName.toLowerCase();
    const cls = attr(el, 'class');
    const type = tag === 'input' ? (el.type || 'text').toLowerCase() : attr(el, 'type').toLowerCase();
    const field = tag === 'input' || tag === 'textarea' || tag === 'select';
    let text = '', value = '';
    if (tag === 'select') text = el.selectedOptions && el.selectedOptions[0] ? norm(el.selectedOptions[0].text) : '';
    else if (tag === 'input' && ['button', 'submit', 'reset'].includes(type)) text = norm(el.value);
    else if (!field) text = textOf(el);
    if ((tag === 'input' && !['password', 'checkbox', 'radio', 'file', 'button', 'submit', 'reset', 'range']
         .includes(type)) || tag === 'textarea') value = norm(el.value);
    const labels = el.labels ? [...el.labels].map(textOf).filter(Boolean).join(' ') : '';
    const lb = attr(el, 'aria-labelledby');
    const labelledby = lb ? lb.split(' ').map(id => document.getElementById(id)).filter(Boolean)
      .map(textOf).join(' ') : '';
    const imgs = [...el.querySelectorAll('img')];
    const icons = [...el.querySelectorAll('svg, i, img, [class*="icon"]')];
    let href = tag === 'a' ? attr(el, 'href').replace(/[?&]sid=[^&#]*/, '') : '';
    const hints = [el.id, cls, attr(el, 'name'), href, ...Object.values(el.dataset || {}),
      ...icons.map(i => i.getAttribute('class') || ''),
      ...[...el.querySelectorAll('use')].map(u => u.getAttribute('href') || u.getAttribute('xlink:href') || ''),
      ...imgs.map(i => (i.getAttribute('src') || '').split('/').pop().split('.')[0])].filter(Boolean).join(' ');
    const ctl = tag === 'label' ? el.control : el;
    const checked = ctl && ctl.tagName === 'INPUT' && ['checkbox', 'radio'].includes(ctl.type) ? ctl.checked : null;
    const lm = el.parentElement && el.parentElement.closest(LANDMARK);
    return {
      tag, type, cls, id: el.id || '', role: attr(el, 'role').toLowerCase(),
      sel: tag + (el.id ? '#' + el.id : '') + (cls ? '.' + cls.split(' ').join('.') : ''),
      text: clip(text, 80), value: clip(value, 60), aria: attr(el, 'aria-label'), title: attr(el, 'title'),
      alt: norm([attr(el, 'alt'), ...imgs.map(i => i.getAttribute('alt') || '')].join(' ')),
      placeholder: attr(el, 'placeholder'), labels: clip(labels, 60), labelledby: clip(labelledby, 60),
      svg_title: clip([...el.querySelectorAll('title')].map(t => t.textContent).join(' '), 60),
      hints: clip(hints, 160), data: clip(Object.values(el.dataset || {}).join(' '), 60),
      context: contextOf(el, text),
      landmark: lm ? norm([lm.tagName.toLowerCase(), attr(lm, 'role'), lm.id, attr(lm, 'class')].join(' ')) : '',
      has_icon: icons.length > 0 || /icon/i.test(cls),
      container: els.some(o => o !== el && el.contains(o)),
      label_for_visible: tag === 'label' && !!el.control && els.includes(el.control),
      in_overlay: overlays.some(o => o.el.contains(el)),
      editable: !!el.isContentEditable,
      checked, ctl_type: ctl && ctl.tagName === 'INPUT' ? ctl.type : '',
      aria_checked: attr(el, 'aria-checked'), aria_pressed: attr(el, 'aria-pressed'),
      aria_selected: attr(el, 'aria-selected'), aria_current: attr(el, 'aria-current'),
      disabled: !!el.disabled || attr(el, 'aria-disabled') === 'true',
      x: pt[0], y: pt[1], x0: b.x0, y0: b.y0, x1: b.x1, y1: b.y1, area: b.w * b.h,
    };
  });
  const headings = [...document.querySelectorAll('h1, h2, h3')]
    .filter(h => visible(h) && box(h)).map(h => clip(textOf(h), 60)).filter(Boolean).slice(0, 8);
  const se = document.scrollingElement || document.documentElement;
  return {
    vw, vh, title: document.title, headings, elements,
    overlays: overlays.map(({kind, text, cx, cy}) => ({kind, text, cx, cy})),
    scroll_y: window.scrollY, more_below: se.scrollHeight > window.scrollY + vh + 4,
  };
}
"""


# --- tokens and matching -------------------------------------------------------------------

_CAMEL = re.compile(r"([a-z])([A-Z])")
_STOPWORDS = {
    "the", "a", "an", "in", "on", "at", "of", "to", "my", "this", "that", "for", "with", "from", "by", "and",
    "or", "it", "its", "is", "near", "beside", "under", "above", "below", "inside", "row", "section",
    "area", "part", "page", "screen", "panel", "side", "click", "tap", "press", "element", "labeled", "labelled",
    "called", "named", "says", "saying", "which", "one", "there", "your", "me",
}
_TYPE_WORDS: dict[str, set[str]] = {
    "button": {"button", "icon button"}, "btn": {"button", "icon button"},
    "icon": {"icon button", "icon"},
    "toggle": {"toggle", "checkbox"}, "switch": {"toggle", "checkbox"},
    "checkbox": {"checkbox", "toggle"}, "check": {"checkbox", "toggle"},
    "link": {"link"},
    "field": {"input", "dropdown"}, "input": {"input"}, "box": {"input", "checkbox", "dropdown"},
    "textbox": {"input"}, "textarea": {"input"},
    "tab": {"tab"},
    "menu": {"menu item", "dropdown"}, "dropdown": {"dropdown", "menu item"}, "option": {"option", "menu item"},
    "radio": {"radio"}, "slider": {"slider"}, "region": {"region"},
}
# Clickable SVG shapes: map regions, chart bars (canvas content, by contrast, is opaque to the DOM).
_SVG_SHAPES = {"path", "rect", "circle", "ellipse", "polygon", "polyline", "g", "text"}
# Relational phrases, removed before tokenizing (so "the Next button" keeps "next").
_RELATIONAL = re.compile(r"\b(next to|close to|on top of)\b", re.I)
_POSITION_WORDS = {"top", "upper", "bottom", "lower", "left", "right", "center", "centre", "middle", "corner"}
_SYNONYM_GROUPS = [
    "gear cog setting settings preference preferences config",
    "bell notification notifications alert alerts notif",
    "magnifier magnifying search find lens",
    "close x dismiss",
    "trash bin delete remove garbage",
    "pencil edit pen",
    "star favorite favourite starred",
    "house home dashboard",
    "envelope mail email message messages inbox",
    "person avatar profile account user",
    "hamburger burger menu",
    "plus add new create",
    "question help faq",
    "toolbar header nav navbar topbar banner menubar navigation",
    "chart graph plot bar",
    "heart like",
    "funnel filter",
    "refresh reload",
    "back previous prev",
    "next forward",
    "eye show view",
    "lock password",
    "info information details",
    "printer print",
    "calendar date",
]


def _stem(t: str) -> str:
    return t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t


def tokens(s: str) -> list[str]:
    """Lowercase alphanumeric tokens, camelCase split, trailing plural 's' dropped."""
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", _CAMEL.sub(r"\1 \2", s or "").lower())]


_SYNONYMS: dict[str, set[str]] = {}
for _group in _SYNONYM_GROUPS:
    _words = {_stem(w) for w in _group.split()}
    for _w in _words:
        _SYNONYMS.setdefault(_w, set()).update(_words - {_w})

# Weights per descriptor group (see module docstring).
W_PRIMARY, W_TEXT, W_CONTAINER_TEXT, W_HINT, W_CONTEXT = 3.0, 3.0, 1.5, 2.0, 1.0
PREFIX, SYNONYM, TYPE_AS_TEXT = 0.8, 0.6, 0.5
TYPE_BONUS, TYPE_BONUS_MAX, POSITION_BONUS, EXACT_BONUS = 1.5, 2.0, 0.75, 1.0


def _match(t: str, toks: set[str]) -> float:
    if t in toks:
        return 1.0
    if len(t) >= 4 and any(len(d) >= 4 and (d.startswith(t) or t.startswith(d)) for d in toks):
        return PREFIX
    if _SYNONYMS.get(t, set()) & toks:
        return SYNONYM
    return 0.0


def kind_of(e: dict) -> str:
    """Coarse element type, shared by the type-word bonus and the summary lines."""
    tag, role, typ = e["tag"], e["role"], e["type"]
    cls = set(tokens(e["cls"]))
    icon_only = e["has_icon"] and not e["text"]
    if tag == "canvas":
        return "canvas"
    if role == "switch" or cls & {"switch", "toggle"}:
        return "toggle"
    if role == "checkbox" or e["ctl_type"] == "checkbox":
        return "checkbox"
    if role == "radio" or e["ctl_type"] == "radio":
        return "radio"
    if role == "slider" or typ == "range":
        return "slider"
    if tag == "select" or role in ("combobox", "listbox"):
        return "dropdown"
    if tag in ("input", "textarea") or role in ("textbox", "searchbox") or e["editable"]:
        return "input"
    if role == "tab":
        return "tab"
    if role.startswith("menuitem"):
        return "menu item"
    if role == "option":
        return "option"
    if tag == "a" or role == "link":
        return "link"
    if tag == "button" or role == "button" or cls & {"btn", "button"}:
        return "icon button" if icon_only else "button"
    if icon_only:
        return "icon"
    if tag in _SVG_SHAPES:
        return "region"
    if tag == "label":
        return "label"
    return "clickable"


def _where(x: float, y: float, vw: float, vh: float) -> str:
    col = 0 if x < vw / 3 else 2 if x > 2 * vw / 3 else 1
    row = 0 if y < vh / 3 else 2 if y > 2 * vh / 3 else 1
    return [["top-left", "top", "top-right"], ["left", "center", "right"],
            ["bottom-left", "bottom", "bottom-right"]][row][col]


def _position_ok(word: str, e: dict, vw: float, vh: float) -> bool:
    cx, cy = (e["x0"] + e["x1"]) / 2, (e["y0"] + e["y1"]) / 2
    col = 0 if cx < vw / 3 else 2 if cx > 2 * vw / 3 else 1
    row = 0 if cy < vh / 3 else 2 if cy > 2 * vh / 3 else 1
    return {"top": row == 0, "upper": row == 0, "bottom": row == 2, "lower": row == 2, "left": col == 0,
            "right": col == 2, "center": col == 1 and row == 1, "centre": col == 1 and row == 1,
            "middle": col == 1, "corner": row != 1 and col != 1}[word]


def score(e: dict, target: str, vw: float, vh: float) -> tuple[float, bool]:
    """(score, eligible) of one candidate for a target description."""
    target_toks = [t for t in dict.fromkeys(tokens(_RELATIONAL.sub(" ", target))) if t not in _STOPWORDS]
    primary = set(tokens(" ".join(e[k] for k in ("aria", "title", "alt", "placeholder", "value", "labels",
                                                  "labelledby", "svg_title"))))
    groups = [
        (primary, W_PRIMARY),
        (set(tokens(e["text"])), W_CONTAINER_TEXT if e["container"] else W_TEXT),
        (set(tokens(e["hints"])), W_HINT),
        (set(tokens(e["context"])) | set(tokens(e["landmark"])), W_CONTEXT),
    ]
    kind = kind_of(e)
    total, content_hit, has_content, type_bonus = 0.0, False, False, 0.0
    matched: set[str] = set()
    for t in target_toks:
        if t in _POSITION_WORDS:
            total += POSITION_BONUS if _position_ok(t, e, vw, vh) else 0.0
            continue
        w = max(_match(t, toks) * weight for toks, weight in groups)
        if t in _TYPE_WORDS:
            w *= TYPE_AS_TEXT
            if kind in _TYPE_WORDS[t] or (t == "icon" and e["has_icon"] and not e["text"]):
                type_bonus += TYPE_BONUS
        else:
            has_content = True
            content_hit = content_hit or w > 0
        if w > 0:
            matched.add(t)
        total += w
    total += min(type_bonus, TYPE_BONUS_MAX)
    name_toks = set(tokens(e["text"])) or set(tokens(e["aria"]))
    if name_toks and name_toks <= matched:
        total += EXACT_BONUS
    eligible = content_hit if has_content else total > 0
    return total, eligible


def rank(elements: list[dict], target: str, vw: float, vh: float) -> list[tuple[float, dict]]:
    """Eligible candidates, best first (higher score, then smaller element)."""
    scored = [(s, e) for e in elements for s, ok in [score(e, target, vw, vh)] if ok and s > 0]
    return sorted(scored, key=lambda se: (-round(se[0], 3), se[1]["area"]))


# --- summary -------------------------------------------------------------------------------

_GENERIC_HINTS = {"icon", "btn", "button", "svg", "img", "image", "fa", "fas", "far", "fab", "material", "symbol",
                  "outlined", "mdi", "bi", "lucide", "feather", "glyph", "ico", "ui", "js", "inline", "small",
                  "large", "sm", "lg", "primary", "secondary", "on", "off", "sel", "selected", "active", "toggle",
                  "switch", "link", "nav", "item", "row", "card", "control"}


def _clip(s: str, n: int) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 3].rstrip() + "..."


def _name(e: dict) -> str:
    return next((e[k] for k in ("text", "aria", "labels", "labelledby", "title", "alt", "svg_title", "placeholder")
                 if e[k]), "")


def _hint_words(e: dict) -> str:
    seen = [t for t in dict.fromkeys(tokens(e["hints"])) if t not in _GENERIC_HINTS and not t.isdigit()
            and (len(t) > 1 or t == "x")]
    return " ".join(seen[:2])


def _describe(e: dict) -> str:
    """Element text, or a short description from id/class/icon hints and the surrounding row."""
    name = _name(e)
    if name:
        desc = _clip(name, 60)
        if e["has_icon"] and not e["text"]:
            desc += " (icon, no visible text)"
        return desc
    hint = _hint_words(e) if e["has_icon"] else ""
    if hint:
        return f"{hint} icon (no text)"
    if e["data"]:  # e.g. a map region's data-name
        return f"{_clip(e['data'], 40)} (no text)"
    return f'(no text) next to "{_clip(e["context"], 40)}"' if e["context"] else "(no text)"


def _is_on(e: dict) -> bool:
    if e["checked"] is not None:
        return bool(e["checked"])
    if e["aria_checked"] or e["aria_pressed"]:
        return "true" in (e["aria_checked"], e["aria_pressed"])
    return bool(set(tokens(e["cls"])) & {"on", "checked", "active", "enabled", "selected"})


def _is_selected(e: dict) -> bool:
    if "true" in (e["aria_selected"], e["aria_pressed"]) or e["aria_current"] not in ("", "false"):
        return True
    return bool(set(tokens(e["cls"])) & {"on", "sel", "selected", "active", "current", "checked"})


def _state_line(e: dict, kind: str) -> str | None:
    label = _clip(_name(e) or e["context"] or _hint_words(e) or kind, 40)
    if kind in ("toggle", "checkbox", "radio"):
        return f"- {label} {kind}: {'on' if _is_on(e) else 'off'}"
    if kind == "input" and e["value"]:
        return f'- {label} field: "{e["value"]}"'
    if kind == "dropdown" and e["text"]:
        label = _clip(e["labels"] or e["aria"] or e["context"] or kind, 40)
        return f'- {label} dropdown: "{_clip(e["text"], 40)}"'
    if _is_selected(e):
        where = f' (in "{_clip(e["context"], 40)}")' if e["context"] and _name(e) else ""
        return f"- {kind} {label}: selected{where}"
    return None


def render_summary(snap: dict, max_lines: int = MAX_LINES) -> str:
    """SUMMARY_PROMPT-shaped text (PAGE / ELEMENTS / STATE / OVERLAYS) from a DOM snapshot."""
    vw, vh = snap["vw"], snap["vh"]
    page = f"PAGE: {snap['title'] or '(untitled)'}"
    if snap["headings"]:
        page += " | headings: " + "; ".join(snap["headings"])
    if snap.get("more_below"):
        page += " | more content below (scroll down)"

    shown = [e for e in snap["elements"] if not e["label_for_visible"]]
    shown.sort(key=lambda e: (not e["in_overlay"], round(e["y0"] / 16), e["x0"]))
    items = [(e, kind_of(e)) for e in shown]
    dupes = Counter((k, _name(e)) for e, k in items)

    elem_lines, state_lines = [], []
    for e, k in items:
        desc = _describe(e)
        if _name(e) and dupes[(k, _name(e))] > 1 and e["context"]:
            desc += f' (in "{_clip(e["context"], 40)}")'
        if k == "canvas":
            desc = (_name(e) or _hint_words(e) or "drawing") + " (canvas: chart or map)"
        if e["disabled"]:
            desc += " (disabled)"
        elem_lines.append(f"- {k}: {desc} @ {_where((e['x0'] + e['x1']) / 2, (e['y0'] + e['y1']) / 2, vw, vh)}")
        if (line := _state_line(e, k)) is not None:
            state_lines.append(line)
    state_lines = state_lines[:10]
    overlay_lines = [f"- {o['kind']}: {o['text']} @ {_where(o['cx'], o['cy'], vw, vh)}"
                     for o in snap["overlays"][:4]]

    fixed = 1 + 1 + (1 + len(state_lines)) + (1 + len(overlay_lines))
    budget = max(1, max_lines - fixed)
    if len(elem_lines) > budget:
        elem_lines = elem_lines[: budget - 1] + [f"- (+{len(elem_lines) - budget + 1} more elements)"]

    lines = [page]
    lines += ["ELEMENTS:"] + elem_lines if elem_lines else ["ELEMENTS: none"]
    lines += ["STATE:"] + state_lines if state_lines else ["STATE: none"]
    lines += ["OVERLAYS:"] + overlay_lines if overlay_lines else ["OVERLAYS: none"]
    return "\n".join(lines[:max_lines])


# --- the oracle ----------------------------------------------------------------------------

def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


class OracleGrounder:
    """TEST-ONLY Grounder + Summarizer that reads the DOM. Never use it in evals or the demo."""

    def __init__(self, browser_session: Any):
        assert_oracle_allowed()
        self.browser_session = browser_session

    @property
    def page(self) -> Any:
        return self.browser_session.page

    async def snapshot(self) -> dict:
        try:
            return await self.page.evaluate(_SNAPSHOT_JS)
        except Exception:  # mid-navigation: the execution context was replaced; try once more
            await asyncio.sleep(0.3)
            return await self.page.evaluate(_SNAPSHOT_JS)

    async def locate(self, img: Image.Image, target: str) -> Located:
        t0 = time.perf_counter()
        try:
            snap = await self.snapshot()
            ranked = rank(snap["elements"], target, snap["vw"], snap["vh"])
        except Exception as e:
            return Located(None, None, error=f"oracle: {type(e).__name__}: {e}", model=MODEL, latency_ms=_ms(t0))
        if not ranked:
            return Located(None, None, error=f"oracle: no element matches {target!r}", model=MODEL,
                           latency_ms=_ms(t0))
        s, e = ranked[0]
        raw = f'oracle: <{e["sel"]} "{_clip(_name(e), 40)}"> score={s:.2f}'
        if e["tag"] == "canvas":
            return Located(None, None, raw=raw, model=MODEL, latency_ms=_ms(t0),
                           error=f"oracle: {target!r} is inside a canvas ({e['sel']}); canvas regions can't be "
                                 "resolved from the DOM")
        return Located(e["x"], e["y"], raw=raw, point_1000=to_grid((e["x"], e["y"]), snap["vw"], snap["vh"]),
                       model=MODEL, latency_ms=_ms(t0))

    async def summarize(self, img: Image.Image) -> Summary:
        t0 = time.perf_counter()
        try:
            text = render_summary(await self.snapshot())
        except Exception as e:
            return Summary("", error=f"oracle: {type(e).__name__}: {e}", model=MODEL, latency_ms=_ms(t0))
        return Summary(text, model=MODEL, latency_ms=_ms(t0))
