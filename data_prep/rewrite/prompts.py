"""Prompt for Lightning instruction rewriting: system prompt, few-shot batch, item format, response schema.

Bump PROMPT_VERSION on any change to SYSTEM_PROMPT, FEW_SHOT or format_batch (the cache is keyed by it).
"""

import json

PROMPT_VERSION = "v4"  # v1: 4-bit plumbing only; v3: v2 audit fixes; v4: positions sent as plain words

STYLES = ("functional", "descriptive", "positional")
DROP_REASONS = ("numeric", "unintelligible", "needs_screen", "duplicate_neighbor")
ITEM_KEYS = ("label", "platform", "control", "kind", "position", "size", "neighbors", "style")
OPTIONAL_KEYS = frozenset({"control", "kind"})
LABEL_MAX_CHARS = 200

SYSTEM_PROMPT = """\
You write click instructions for training a GUI grounding model. Each item is one on-screen element, the target, \
with its raw accessibility or HTML label. Turn the label into a short, natural instruction for clicking it, like \
the ScreenSpot benchmark ("open the settings", "the Recent tab"). You cannot see the screen: use only the \
item's fields.

Item fields:
- label: the target's raw label.
- platform: mobile, web or desktop.
- control: plain noun for the element type, if known (button, link, tab, text field, checkbox, menu item, \
dropdown, text, area, element...). kind: text or icon, if known.
- position: where the target's center is on a 3x3 screen grid (top left, top center, ..., center, ..., bottom \
right). Say it naturally: "at the top left", "on the right", "in the middle".
- size: small, medium or large, relative to the screen.
- neighbors: other labeled elements on the same screen, nearest first; not necessarily adjacent. target_is: where \
the target is relative to that neighbor; copy it as is: target_is right of Search -> "right of Search".
- style: the style to write.

Styles (examples are label -> instruction):
- functional: an imperative action, verb from the label's meaning: More options -> "open more options", Email \
address (text field) -> "enter the email address". Unsure what the click does: a neutral verb (click, tap, \
select). On/off state is unknown: "toggle Bluetooth", not "turn on Bluetooth".
- descriptive: a noun phrase naming the target: Settings (button) -> "the Settings button", Recent (tab) -> "the \
Recent tab", Battery (text) -> "the Battery text".
- positional: locate the target by its position and/or a neighbor, keeping a word of its label: "the Share \
button at the top right", "the Save button below Title". Prefer an exact target_is relation to "next to" or "near".

Rules:
1. 3-8 words, one line, no final period, no quotation marks. Short labels: add the control noun or a neutral \
verb ("click the Copy button"), not invented context.
2. Faithful: item k's instruction is about item k's own label and keeps at least one word of it, in every style. \
Never describe or name a neighbor instead; neighbors only help locate the target.
3. Never invent visual details (colors, icon shapes such as arrow, gear, star or magnifying glass, styling) \
unless the word is in the label or a neighbor label. Size words: only "small" or "large", when size says so.
4. Never invent text or purpose: every word comes from the label, a neighbor label, control, the position or a \
plain verb. Search -> "click Search", not a made-up query.
5. Copy proper nouns, brands, acronyms, mixed-case words and visible names (files, domains, usernames) exactly \
(GitHub, Wi-Fi, PDF, iPhone, output_20240911_164440).
6. Element nouns (button, link, icon, tab, field, toggle...) come only from control or the label. control text or \
area: the box is the text or region itself, not a button or toggle beside it: never call it a button, link, icon, \
tab or toggle; functional style uses only click, tap or select ("tap Digital Wellbeing"). control element or \
missing (web pages): no element noun, at most a neutral item, entry or option.
7. Long labels (sentences, headlines, link texts, accessibility descriptions): in any style, a short phrase naming \
the thing in its own key words, never the copied sentence: "Bar Chart. Showing App usage data with 7 data \
points." -> "the app usage bar chart".
8. Look-alikes: if a neighbor label is identical or starts with the same word ("Download App" vs "Download PDF"), \
add what tells the target apart: its distinct words ("download the app"), a target_is relation or its position.
9. style only says how to phrase it: never write the words functional, descriptive or positional.
10. Non-English label: keep its words verbatim inside an English instruction.

Drop an item (keep=false) only when no faithful instruction is possible, with one drop_reason:
- numeric: only numbers, sizes, prices, dates, times, percentages, counters or relative times, even with a unit or \
status word ("Updated 2 hrs ago · 4.5 MB").
- unintelligible: not readable words: resource or bus IDs (":1.72/StatusNotifierItem"), AutomationID values, \
code names (btn_submit_2), long URLs or paths, gibberish, single characters or glyphs.
- needs_screen: too generic to identify the element, and nothing in the item pins it down ("Image", "Icon", \
"Button", "View", "Item 3").
- duplicate_neighbor: a neighbor has the same label (ignoring case and punctuation) and no target_is relation or \
position tells them apart.

Do not over-drop: most items must be kept. Ordinary short labels ("Search", "Close", "Sign in", a name or a \
title) and long text (shorten it) are fine. Terse is not unintelligible.

Output exactly one JSON object and nothing else (no prose, no markdown):
{"items":[{"k":1,"keep":true,"instruction":"<instruction>"},{"k":2,"keep":false,"drop_reason":"<drop_reason>"}]}
One item per input k, in order. Kept items have "instruction"; dropped items have "drop_reason"."""


def _nb(*pairs):
    return [{"label": label, "target_is": target_is} for label, target_is in pairs]


# One realistic batch (6 keeps, 3 drops) from our sources: Android, GNOME, macOS System Settings, web pages. The keeps
# show the v2 audit fixes: a text/area control, a long description, a look-alike, target_is copied, no invented noun.
FEW_SHOT_METAS = [
    {"label": "Digital Wellbeing", "platform": "mobile", "control": "text", "position": "middle-left",
     "size": "small", "neighbors": _nb(("Screen time", "below"), ("Bedtime mode", "above"), ("Focus mode", "above")),
     "style": "functional"},
    {"label": "Updated 2 hrs ago · 4.5 MB", "platform": "mobile", "control": "text", "position": "top-left",
     "size": "small", "neighbors": _nb(("report_final.pdf", "below"), ("Recent files", "below"), ("Sort", "left of")),
     "style": "descriptive"},
    {"label": "Search", "platform": "desktop", "control": "button", "position": "top-right", "size": "small",
     "neighbors": _nb(("Main Menu", "left of"), ("Files", "right of"), ("Close", "left of")), "style": "positional"},
    {"label": "Bar Chart. Showing App usage data with 7 data points.", "platform": "mobile", "control": "area",
     "position": "center", "size": "large", "neighbors": _nb(("Today", "below"), ("Instagram", "above"),
                                                             ("Chrome", "above")), "style": "functional"},
    {"label": ":1.72/StatusNotifierItem", "platform": "desktop", "control": "menu", "position": "top-right",
     "size": "small", "neighbors": _nb(("System", "left of"), ("Firefox Web Browser", "right of")),
     "style": "functional"},
    {"label": "Download Best Share Market App", "platform": "web", "position": "middle-right", "size": "small",
     "neighbors": _nb(("Download Stock Market Courses", "below"), ("Share Market Tips", "above"),
                      ("Download Trading Guide", "above")), "style": "descriptive"},
    {"label": "Privacy & Security", "platform": "desktop", "control": "text", "position": "middle-left",
     "size": "small", "neighbors": _nb(("Siri & Spotlight", "below"), ("Desktop & Dock", "above"),
                                       ("Touch ID & Password", "above")), "style": "descriptive"},
    {"label": "Image", "platform": "web", "position": "top-left", "size": "medium",
     "neighbors": _nb(("Skip to content", "below"), ("Home", "left of"), ("About Us", "left of")),
     "style": "descriptive"},
    {"label": "Contact Us", "platform": "web", "position": "bottom-right", "size": "small",
     "neighbors": _nb(("Client Handbook", "right of"), ("FAQs", "right of"), ("Privacy Policy", "below")),
     "style": "positional"},
]

FEW_SHOT_ANSWER = {"items": [
    {"k": 1, "keep": True, "instruction": "tap Digital Wellbeing"},
    {"k": 2, "keep": False, "drop_reason": "numeric"},
    {"k": 3, "keep": True, "instruction": "the Search button left of Main Menu"},
    {"k": 4, "keep": True, "instruction": "tap the app usage bar chart"},
    {"k": 5, "keep": False, "drop_reason": "unintelligible"},
    {"k": 6, "keep": True, "instruction": "the Best Share Market App download"},
    {"k": 7, "keep": True, "instruction": "the Privacy & Security text"},
    {"k": 8, "keep": False, "drop_reason": "needs_screen"},
    {"k": 9, "keep": True, "instruction": "Contact Us at the bottom right"},
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
        elif key == "position":
            value = str(value).replace("-", " ")  # "top right", not the "top-right" token (v3 copied it verbatim)
        elif key == "neighbors":
            value = [{"label": n["label"], "target_is": n["target_is"]} for n in value or []]
        item[key] = value
    return _dumps(item)


def format_batch(metas: list[dict]) -> str:
    """User-message text: a short reminder plus one item line per meta, k = 1..N."""
    n = len(metas)
    count = "this item" if n == 1 else f"these {n} items"
    head = (f"Rewrite {count}. Reply with one JSON object {{\"items\":[...]}} holding exactly {n} item"
            f"{'' if n == 1 else 's'}, k = 1 to {n}, in order. Follow each item's style and keep a word of its own "
            "label. A neighbor's target_is is where the target is relative to that neighbor: copy it as is. Never "
            "invent colors, shapes, text or element types.")
    return head + "\n\n" + "\n".join(format_item(k, m) for k, m in enumerate(metas, 1))


FEW_SHOT = [
    {"role": "user", "content": format_batch(FEW_SHOT_METAS)},
    {"role": "assistant", "content": _dumps(FEW_SHOT_ANSWER)},
]


def build_messages(metas: list[dict]) -> list[dict]:
    """Chat messages for one request: system prompt, few-shot pair(s), then the batch."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, *(dict(m) for m in FEW_SHOT),
            {"role": "user", "content": format_batch(metas)}]
