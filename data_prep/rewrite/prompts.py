"""Prompt for Lightning instruction rewriting: system prompt, few-shot batch, item format, response schema.

Bump PROMPT_VERSION on any change to SYSTEM_PROMPT, FEW_SHOT or format_batch (the cache is keyed by it).
"""

import json

PROMPT_VERSION = "v2"  # v1 only ever ran for plumbing on a community 4-bit build

STYLES = ("functional", "descriptive", "positional")
DROP_REASONS = ("numeric", "unintelligible", "needs_screen", "duplicate_neighbor")
ITEM_KEYS = ("label", "platform", "control", "kind", "position", "size", "neighbors", "style")
OPTIONAL_KEYS = frozenset({"control", "kind"})
LABEL_MAX_CHARS = 200

SYSTEM_PROMPT = """\
You write click instructions for training a GUI grounding model. Each input item is one on-screen element with \
its raw accessibility or HTML label. Turn the label into a short, natural instruction a user would give to click \
that element, like the ScreenSpot benchmark ("open the settings", "the search bar", "close the window"). You \
cannot see the screen: use only the item's fields.

Item fields:
- label: the element's raw label.
- platform: mobile, web or desktop.
- control: plain noun for the element type, if known (button, link, tab, text field, checkbox, menu item, \
dropdown, tree item, table cell, text, area, element...). kind: text or icon, if known.
- position: where the element's center is on a 3x3 screen grid (top-left, top-center, ..., center, ..., \
bottom-right).
- size: small, medium or large, relative to the screen.
- neighbors: other labeled elements on the same screen, nearest first; not necessarily adjacent. direction says \
where THAT NEIGHBOR is relative to the target: {"label":"Search","direction":"left"} means Search is left of the \
target, so the target is RIGHT of Search. Likewise "right" = target is left of it, "above" = target is below it, \
"below" = target is above it, "overlapping" = no direction.
- style: the style to write.

Styles:
- functional: an imperative action. "open the settings", "go back", "search for hotels", "enter the email \
address". Pick the verb from the label's meaning; if unsure what the click does, use a neutral verb (open, select, \
tap). On/off state is unknown: "toggle Bluetooth", not "turn on Bluetooth".
- descriptive: a noun phrase naming the element. "the Settings button", "the Recent tab", "the password field".
- positional: locate the element by its position and/or neighbors. "the Share icon at the top right", "the field \
below Password", "the tab between Home and Library". Keep the label's key word unless the location is clearly \
unique. Say positions plainly ("at the top", "on the left", "in the middle"). Prefer exact directions ("right of \
Search") over "next to" or "near".

Rules:
1. 3-8 words, one line, no final period, no quotation marks. Lowercase is fine. Short labels: add the control \
noun or a neutral verb ("click the Copy button", "select Documents") rather than inventing context.
2. Faithful: describe the target, the element named by label, using only facts from the item. Never describe or \
name a neighbor instead; neighbors only help locate the target. Write the requested style.
3. Never invent visual details: no colors, icon shapes (arrow, gear, star, heart, plus, magnifying glass...) \
or styling unless that word is in the label or a neighbor label. Size words: only "small" or "large", \
only when size says so.
4. Never invent text: any on-screen text you mention comes from the label or a neighbor label.
5. Copy proper nouns, brands, product names, acronyms and mixed-case words exactly (GitHub, Wi-Fi, PDF, iPhone).
6. control is the only noun you may use for the element type ("the Search button", "the Title text field"). \
text and area are labels or regions, not buttons: say "the text labeled X", "the area labeled X", or just name X \
("open Privacy & Security"). If control is element or missing, use no noun or a neutral one (option, item, \
entry); never guess button, icon or link.
7. Long labels (sentences, headlines, link texts): keep only the words that identify the element, verbatim. \
"Learn more about our free shipping policy" -> "read about free shipping".
8. Non-English label: keep its words verbatim inside an English instruction.
9. Keep visible names verbatim: file names, domains, usernames ("open output_20240911_164440").

Drop an item (keep=false) only when no faithful instruction is possible, with one drop_reason:
- numeric: only a number, price, date, time, percentage, counter or relative time ("12:45 PM", "$4.99", \
"Oct 12", "73 minutes ago").
- unintelligible: not readable words: resource or bus IDs (":1.72/StatusNotifierItem"), AutomationID values, \
code or variable names (btn_submit_2), long URLs or file paths, gibberish, single characters or glyphs.
- needs_screen: too generic to identify the element, and nothing in the item pins it down ("Image", "Icon", \
"Button", "View", "Item 3").
- duplicate_neighbor: a neighbor has the same or nearly the same label (ignoring case and punctuation), so the \
instruction would be ambiguous. Exception: if style is positional and the directions tell them apart, keep it and \
write a positional instruction that does.

Do not over-drop: most items must be kept. Ordinary short labels ("Search", "Close", "More options", "Sign in", \
"Next", a name or a title) and long text (shorten it) are fine. Terse is not unintelligible.

Output exactly one JSON object and nothing else (no prose, no markdown):
{"items":[{"k":1,"keep":true,"instruction":"<instruction>"},{"k":2,"keep":false,"drop_reason":"<drop_reason>"}]}
One item per input k, in order. Kept items have "instruction"; dropped items have "drop_reason"."""


def _nb(*pairs):
    return [{"label": label, "direction": direction} for label, direction in pairs]


# One realistic batch (6 keeps, 2 drops) from our sources: Linux GNOME, macOS System Settings, Android, a web page.
FEW_SHOT_METAS = [
    {"label": "Firefox Web Browser", "platform": "desktop", "control": "button", "position": "top-left",
     "size": "small", "neighbors": _nb(("Rhythmbox", "below"), ("Help", "below"), ("Show Applications", "below")),
     "style": "functional"},
    {"label": "73 minutes ago", "platform": "mobile", "control": "text", "position": "top-right", "size": "small",
     "neighbors": _nb(("Inbox", "left"), ("Doctor's appointment", "below"), ("Search mail", "above")),
     "style": "functional"},
    {"label": "Privacy & Security", "platform": "desktop", "control": "text", "position": "middle-left",
     "size": "small", "neighbors": _nb(("Siri & Spotlight", "above"), ("Desktop & Dock", "above"),
                                       ("Touch ID & Password", "below")), "style": "functional"},
    {"label": "Title", "platform": "mobile", "control": "text field", "position": "top-center", "size": "medium",
     "neighbors": _nb(("Save", "above"), ("Cancel", "above"), ("Description", "below")), "style": "descriptive"},
    {"label": "Aceptar todo", "platform": "web", "position": "bottom-center", "size": "small",
     "neighbors": _nb(("Rechazar todo", "left"), ("Configurar", "left"), ("Política de cookies", "above")),
     "style": "functional"},
    {"label": ":1.72/StatusNotifierItem", "platform": "desktop", "control": "menu", "position": "top-right",
     "size": "small", "neighbors": _nb(("System", "right"), ("Firefox Web Browser", "left")),
     "style": "descriptive"},
    {"label": "Search", "platform": "desktop", "control": "button", "position": "top-right", "size": "small",
     "neighbors": _nb(("Main Menu", "right"), ("Files", "left"), ("Close", "right")), "style": "positional"},
    {"label": "Doctor's appointment", "platform": "mobile", "control": "area", "position": "center",
     "size": "medium", "neighbors": _nb(("Start Wednesday, October 4", "below"), ("Edit", "above")),
     "style": "descriptive"},
]

FEW_SHOT_ANSWER = {"items": [
    {"k": 1, "keep": True, "instruction": "open Firefox Web Browser"},
    {"k": 2, "keep": False, "drop_reason": "numeric"},
    {"k": 3, "keep": True, "instruction": "open Privacy & Security"},
    {"k": 4, "keep": True, "instruction": "the Title text field"},
    {"k": 5, "keep": True, "instruction": "click Aceptar todo"},
    {"k": 6, "keep": False, "drop_reason": "unintelligible"},
    {"k": 7, "keep": True, "instruction": "the Search button left of Main Menu"},
    {"k": 8, "keep": True, "instruction": "the area labeled Doctor's appointment"},
]}

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {"k": {"type": "integer"}, "keep": {"type": "boolean"},
                       "instruction": {"type": ["string", "null"]},
                       "drop_reason": {"type": ["string", "null"], "enum": [*DROP_REASONS, None]}},
        "required": ["k", "keep"]}}},
    "required": ["items"],
}


def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def format_item(k: int, meta: dict) -> str:
    """One compact JSON line for item k; optional keys (control, kind) are omitted when absent."""
    item = {"k": k}
    for key in ITEM_KEYS:
        value = meta.get(key)
        if value is None and key in OPTIONAL_KEYS:
            continue
        if key == "label":
            value = " ".join(str(value).split())
            if len(value) > LABEL_MAX_CHARS:
                value = value[:LABEL_MAX_CHARS].rstrip() + "…"
        elif key == "neighbors":
            value = [{"label": n["label"], "direction": n["direction"]} for n in value or []]
        item[key] = value
    return _dumps(item)


def format_batch(metas: list[dict]) -> str:
    """User-message text: a short reminder plus one item line per meta, k = 1..N."""
    n = len(metas)
    count = "this item" if n == 1 else f"these {n} items"
    head = (f"Rewrite {count}. Reply with one JSON object {{\"items\":[...]}} holding exactly {n} item"
            f"{'' if n == 1 else 's'}, k = 1 to {n}, in order. Follow each item's style. A neighbor's direction is where "
            "that neighbor is relative to the target. Never invent colors, shapes or text.")
    return head + "\n\n" + "\n".join(format_item(k, m) for k, m in enumerate(metas, 1))


FEW_SHOT = [
    {"role": "user", "content": format_batch(FEW_SHOT_METAS)},
    {"role": "assistant", "content": _dumps(FEW_SHOT_ANSWER)},
]


def build_messages(metas: list[dict]) -> list[dict]:
    """Chat messages for one request: system prompt, few-shot pair(s), then the batch."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, *(dict(m) for m in FEW_SHOT),
            {"role": "user", "content": format_batch(metas)}]
