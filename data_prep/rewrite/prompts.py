"""Prompt for Lightning instruction rewriting: system prompt, few-shot batch, item format, response schema.

Bump PROMPT_VERSION on any change to SYSTEM_PROMPT, FEW_SHOT or format_batch (the cache is keyed by it).
"""

import json

PROMPT_VERSION = "v1"

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
- control: UI control type, if known. kind: text or icon, if known.
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
1. 3-8 words, one line, no final period, no quotation marks. Lowercase is fine.
2. Faithful: describe this element only, using only facts from the item.
3. Never invent visual details: no colors, icon shapes (arrow, gear, star, heart, plus, magnifying glass...) \
or styling unless that word is in the label or a neighbor label. Size words: only "small" or "large", \
only when size says so.
4. Never invent text: any on-screen text you mention comes from the label or a neighbor label.
5. Copy proper nouns, brands, product names, acronyms and mixed-case words exactly (GitHub, Wi-Fi, PDF, iPhone).
6. Use control only to pick a generic noun (ImageButton = button or icon, Hyperlink = link, TabItem = tab, \
EditText/Edit = field, Switch = toggle, CheckBox = checkbox, Spinner/ComboBox = dropdown, MenuItem = menu item, \
ListItem = item). If control is missing or generic (View, Layout, Pane, Group, Custom), say option, item, entry or \
no noun.
7. Long labels (sentences, headlines, link texts): keep only the words that identify the element, verbatim. \
"Learn more about our free shipping policy" -> "read about free shipping".
8. Non-English label: keep its words verbatim inside an English instruction.

Drop an item (keep=false) only when no faithful instruction is possible, with one drop_reason:
- numeric: only a number, price, date, time, percentage or counter ("12:45 PM", "$4.99", "Oct 12").
- unintelligible: not readable words: resource IDs, AutomationID values, code or variable names (btn_submit_2), \
URLs, file paths, gibberish, single characters or glyphs.
- needs_screen: too generic to identify the element, and nothing in the item pins it down ("Image", "Icon", \
"Button", "View", "Item 3").
- duplicate_neighbor: a neighbor has the same or nearly the same label (ignoring case and punctuation), so the \
instruction would be ambiguous. Exception: if style is positional and the directions tell them apart, keep it and \
write a positional instruction that does.

Do not over-drop: most items must be kept. Ordinary short labels are fine ("Search", "Close", "More options", \
"Sign in", "Read more", "Next", a name or a title), and long text is fine (shorten it). Short or terse is not \
unintelligible. The example covers every drop reason, but drops are rare in real data.

Output exactly one JSON object and nothing else (no prose, no markdown):
{"items":[{"k":1,"keep":true,"instruction":"open more options"},{"k":2,"keep":false,"drop_reason":"numeric"}]}
One item per input k, in order. Kept items have "instruction"; dropped items have "drop_reason"."""


def _nb(*pairs):
    return [{"label": label, "direction": direction} for label, direction in pairs]


# One realistic batch: Android Settings, an Android music app, two web pages, Windows Notepad.
FEW_SHOT_METAS = [
    {"label": "Wi-Fi", "platform": "mobile", "control": "Switch", "position": "top-right", "size": "small",
     "neighbors": _nb(("Network & internet", "left"), ("Airplane mode", "left"), ("Navigate up", "left"),
                      ("12:45 PM", "left")), "style": "descriptive"},
    {"label": "12:45 PM", "platform": "mobile", "control": "TextView", "position": "top-left", "size": "small",
     "neighbors": _nb(("Navigate up", "below"), ("Network & internet", "below"), ("Airplane mode", "below"),
                      ("Wi-Fi", "right")), "style": "functional"},
    {"label": "Image", "platform": "mobile", "control": "ImageView", "position": "center", "size": "large",
     "neighbors": _nb(("Play", "below"), ("Library", "below"), ("Home", "below"), ("Search", "below")),
     "style": "descriptive"},
    {"label": "Library", "platform": "mobile", "control": "Button", "position": "bottom-center", "size": "small",
     "neighbors": _nb(("Home", "left"), ("Search", "right"), ("Play", "above"), ("Image", "above")),
     "style": "positional"},
    {"label": "Read the full 2024 Annual Report on Ocean Conservation", "platform": "web", "position": "center",
     "size": "medium", "neighbors": _nb(("Our Work", "above"), ("Contact", "above"), ("About Us", "above"),
                                        ("Donate now", "right")), "style": "functional"},
    {"label": "Add to Cart", "platform": "web", "position": "middle-right", "size": "small",
     "neighbors": _nb(("Size guide", "left"), ("Trail Runner 2 - Men's", "above"), ("Add to cart", "below")),
     "style": "descriptive"},
    {"label": "AutomationID: RichEditD2DPT", "platform": "desktop", "control": "Edit", "position": "center",
     "size": "large", "neighbors": _nb(("Minimize", "right"), ("Untitled - Notepad", "left"), ("Maximize", "right"),
                                       ("Close", "right")), "style": "positional"},
    {"label": "Minimize", "platform": "desktop", "control": "Button", "position": "top-right", "size": "small",
     "neighbors": _nb(("Maximize", "right"), ("Close", "right"), ("AutomationID: RichEditD2DPT", "left"),
                      ("Untitled - Notepad", "left")), "style": "functional"},
]

FEW_SHOT_ANSWER = {"items": [
    {"k": 1, "keep": True, "instruction": "the Wi-Fi toggle"},
    {"k": 2, "keep": False, "drop_reason": "numeric"},
    {"k": 3, "keep": False, "drop_reason": "needs_screen"},
    {"k": 4, "keep": True, "instruction": "the bottom button between Home and Search"},
    {"k": 5, "keep": True, "instruction": "open the 2024 Annual Report"},
    {"k": 6, "keep": False, "drop_reason": "duplicate_neighbor"},
    {"k": 7, "keep": False, "drop_reason": "unintelligible"},
    {"k": 8, "keep": True, "instruction": "minimize the window"},
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
