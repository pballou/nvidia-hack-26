# Lightning instruction rewrites: method (prompt v4)

Written by `data_prep/rewrite/rewrite.py` from `stats.json` and the prompt in `data_prep/rewrite/prompts.py`, so it always matches the run it describes. Files: `rewrites.jsonl` (one row per target), `stats.json` (all counts), this file.

## Purpose

OS-Atlas Linux, macOS, AndroidWorld, FineWeb and Windows "instructions" are raw accessibility or HTML labels ("More options", "System"). ScreenSpot-v2 and real users phrase targets as short instructions ("open the settings"). NVIDIA Nemotron 3.5 Lightning rewrites a deterministic subset of those raw labels into short, faithful click instructions in three styles, to train a small (<= 3B) GUI grounding VLM. The other raw rows keep their label, and natural-instruction sources pass through unchanged. `apply.py` turns this file's `rewrites.jsonl` into `train_lightning.jsonl` (mixed) and `train_template.jsonl` (the same rows and order with raw labels) for the ablation, written next to the manifest.

## Model and request settings

Who generated the rows, from the cached records (one value per row; every value is listed when rows were generated in several invocations):

|  | rows |
| --- | --- |
| model (recorded in rewrites.jsonl) | `NVIDIA-hosted API, nvidia/nemotron-3.5-lightning-30b-a3b` (8028 rows); `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (1003 rows) |
| served checkpoint (`GET /models` root; `(not reported)` = NVIDIA's hosted API, which reports none) | `(not reported)` (8028 rows); `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (1003 rows) |
| endpoint | `hosted` (8028 rows); `local` (1003 rows) |
| dropped by rule, no model | 353 rows (`model` = `rule:is_numeric_label`) |

The invocation that wrote this file (it may have sent nothing, e.g. a resume):

| setting | value |
| --- | --- |
| model label | `NVIDIA-hosted API, nvidia/nemotron-3.5-lightning-30b-a3b` |
| model name sent | `nvidia/nemotron-3.5-lightning-30b-a3b` |
| model reported by the server | n/a |
| served checkpoint (`GET /models` root for that name) | not reported (NVIDIA's hosted API reports no root) |
| endpoint | hosted (`https://integrate.api.nvidia.com/v1`), OpenAI-compatible chat completions |
| requests sent | 0 |
| response_format | requested `json_object`, final `json_object` |
| rows per request / concurrent requests | 20 / 16 |
| started / finished (UTC) | 2026-10-17T22:17:03Z / 2026-10-17T22:17:04Z |
| splits rewritten | train |

Note: Generated 2026-10-17 (US Eastern evening) by two endpoints serving NVIDIA Nemotron 3.5 Lightning 30B-A3B: the NVIDIA-hosted API (build.nvidia.com, model nvidia/nemotron-3.5-lightning-30b-a3b) and, after hosted rate-limiting at ~18:00, the official local checkpoint nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 (vllm-metal on a Mac, also used by the agent and demo). Each row's endpoint and served checkpoint are in the cache; counts are in the provenance table.

Provenance check: before sending anything to a local server, `rewrite.py` reads `GET /models` and refuses to run unless the served checkpoint is an `nvidia/` id with no community or quantization marker (`mlx-community`, `4bit`, `8bit`, `gguf`, `awq`, `gptq`, `-q4`, `-q8`); the reported data must come from `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` or NVIDIA's hosted endpoint. Each cached row records the checkpoint that produced it.

Sampling parameters sent with every request. Reasoning ("thinking") is off via `chat_template_kwargs.enable_thinking=false`:

```json
{
  "temperature": 0.7,
  "top_p": 0.95,
  "max_tokens": 4096,
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

## Which rows are rewritten

Raw-label sources: `os_atlas/androidworld`, `os_atlas/fineweb`, `os_atlas/linux`, `os_atlas/macos`, `os_atlas/windows`. Natural-instruction sources (passed through unchanged): `groundcua`, `os_atlas/uibert`, `os_atlas/widget_captioning`. A raw-label row is selected for rewriting if and only if:

```python
int(hashlib.sha256(row["id"].encode("utf-8")).hexdigest(), 16) % 100 < 60
```

Unselected raw-label rows keep their raw label (`instruction_style` = `raw` after `apply.py`).

## Numeric labels: dropped by rule, never sent

Before batching, a selected row whose label (normalized as in `label` below) satisfies `validate.is_numeric_label` is recorded as `keep=false`, `drop_reason=numeric`, `model=rule:is_numeric_label`, `attempts=0` and is never sent to Lightning (numbers, prices, percentages, counters, clock times, dates, relative times). 353 rows in this file's counts were dropped this way. The rule, verbatim:

```python
def is_numeric_label(label: str) -> bool:
    """True if the whole label (normalized, case-insensitive) is only numbers, prices, percentages, counters, sizes,
    durations, clock times, dates or relative times: "42", "$4.99", "12:45 PM", "13 Oct 2023 10:00 AM",
    "73 minutes ago", "now", "(12)". Labels with any other word ("Wi-Fi 6", "Page 2", "Today") are not."""
    s = metalib.normalize_label(label).strip().rstrip(".,;:!").strip()
    return 0 < len(s) <= NUMERIC_LABEL_MAX_CHARS and NUMERIC_LABEL_RE.fullmatch(s) is not None

NUMERIC_LABEL_MAX_CHARS = 64
```

`validate.NUMERIC_LABEL_RE` (flags: `re.IGNORECASE`), the regex text verbatim:

```text
(?:(?:\d{4}-\d{1,2}-\d{1,2}(?:t\d{1,2}:\d{2}(?::\d{2})?(?:z|[+\-]\d{2}:?\d{2})?)?|\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}|\d{4} (?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.? \d{1,2}(?:st|nd|rd|th)?\.?|\d{1,2}(?:st|nd|rd|th)?\.? (?:mon|tues?|wed(?:nes)?|thu(?:rs?)?|fri|sat(?:ur)?|sun)(?:day)?\.?|(?:(?:today|yesterday|tomorrow|(?:mon|tues?|wed(?:nes)?|thu(?:rs?)?|fri|sat(?:ur)?|sun)(?:day)?\.?),? )?(?:(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.? \d{1,2}(?:st|nd|rd|th)?\.?|\d{1,2}(?:st|nd|rd|th)?\.? (?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?)(?:,? (?:\d{4}|'\d{2}))?|(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?,? (?:\d{4}|'\d{2}))(?:,? (?:at )?(?:\d{1,2}(?::\d{2}){1,2}(?: ?[ap]\.?m\.?)?|\d{1,2} ?[ap]\.?m\.?))?|(?:(?:today|yesterday|tomorrow|(?:mon|tues?|wed(?:nes)?|thu(?:rs?)?|fri|sat(?:ur)?|sun)(?:day)?\.?),? (?:at )?)?(?:\d{1,2}(?::\d{2}){1,2}(?: ?[ap]\.?m\.?)?|\d{1,2} ?[ap]\.?m\.?)|(?:[+\-−]?\d+(?:[.,]\d+)*|an?|a few) ?(?:s|secs?|seconds?|m|mins?|minutes?|h|hrs?|hours?|d|days?|w|wks?|weeks?|mos?|months?|y|yrs?|years?)\.? (?:ago|left|remaining)|in [+\-−]?\d+(?:[.,]\d+)* ?(?:s|secs?|seconds?|m|mins?|minutes?|h|hrs?|hours?|d|days?|w|wks?|weeks?|mos?|months?|y|yrs?|years?)\.?|(?:just |right )?now|(?:a )?moments? ago|[+\-−]?\d+(?:[.,]\d+)* ?(?:secs?|seconds?|mins?|minutes?|hrs?|hours?|days?|wks?|weeks?|months?|yrs?|years?)\.?(?: read)?|[+\-−]?\d+(?:[.,]\d+)* ?[hmsw]|[+\-−]?\d+(?:[.,]\d+)* ?(?:[kmgt]i?b|bytes?|[kmg]?hz|[kmg]bps|km|mi|miles?|ft|cm|mm|kg|lbs?|oz|pt|px|ms|mph|°[cf]?)|[+\-−]?\d+(?:[.,]\d+)* ?° ?[nsew]?|[+\-−]?\d+(?:[.,]\d+)* ?%|[+\-−]?(?:[$€£¥₹₩]|us\$|usd|eur|gbp) ?[+\-−]?\d+(?:[.,]\d+)*|[+\-−]?\d+(?:[.,]\d+)* ?(?:[$€£¥₹₩]|usd|eur|gbp)|(?:\d{2,}(?:[.,]\d+)?|\d+[.,]\d+) ?k|[+\-−]?\d+(?:[.,]\d+)*\+?|\([+\-−]?\d+(?:[.,]\d+)*(?: ?(?:[kmgt]i?b|bytes?|[kmg]?hz|[kmg]bps|km|mi|miles?|ft|cm|mm|kg|lbs?|oz|pt|px|ms|mph|°[cf]?))?\)|\[[+\-−]?\d+(?:[.,]\d+)*\]|\((?:mon|tues?|wed(?:nes)?|thu(?:rs?)?|fri|sat(?:ur)?|sun)(?:day)?\.?\))(?:(?:,? | ?[-–—/·|] ?| of )(?:(?:\d{4}-\d{1,2}-\d{1,2}(?:t\d{1,2}:\d{2}(?::\d{2})?(?:z|[+\-]\d{2}:?\d{2})?)?|\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}|\d{4} (?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.? \d{1,2}(?:st|nd|rd|th)?\.?|\d{1,2}(?:st|nd|rd|th)?\.? (?:mon|tues?|wed(?:nes)?|thu(?:rs?)?|fri|sat(?:ur)?|sun)(?:day)?\.?|(?:(?:today|yesterday|tomorrow|(?:mon|tues?|wed(?:nes)?|thu(?:rs?)?|fri|sat(?:ur)?|sun)(?:day)?\.?),? )?(?:(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.? \d{1,2}(?:st|nd|rd|th)?\.?|\d{1,2}(?:st|nd|rd|th)?\.? (?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?)(?:,? (?:\d{4}|'\d{2}))?|(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?,? (?:\d{4}|'\d{2}))(?:,? (?:at )?(?:\d{1,2}(?::\d{2}){1,2}(?: ?[ap]\.?m\.?)?|\d{1,2} ?[ap]\.?m\.?))?|(?:(?:today|yesterday|tomorrow|(?:mon|tues?|wed(?:nes)?|thu(?:rs?)?|fri|sat(?:ur)?|sun)(?:day)?\.?),? (?:at )?)?(?:\d{1,2}(?::\d{2}){1,2}(?: ?[ap]\.?m\.?)?|\d{1,2} ?[ap]\.?m\.?)|(?:[+\-−]?\d+(?:[.,]\d+)*|an?|a few) ?(?:s|secs?|seconds?|m|mins?|minutes?|h|hrs?|hours?|d|days?|w|wks?|weeks?|mos?|months?|y|yrs?|years?)\.? (?:ago|left|remaining)|in [+\-−]?\d+(?:[.,]\d+)* ?(?:s|secs?|seconds?|m|mins?|minutes?|h|hrs?|hours?|d|days?|w|wks?|weeks?|mos?|months?|y|yrs?|years?)\.?|(?:just |right )?now|(?:a )?moments? ago|[+\-−]?\d+(?:[.,]\d+)* ?(?:secs?|seconds?|mins?|minutes?|hrs?|hours?|days?|wks?|weeks?|months?|yrs?|years?)\.?(?: read)?|[+\-−]?\d+(?:[.,]\d+)* ?[hmsw]|[+\-−]?\d+(?:[.,]\d+)* ?(?:[kmgt]i?b|bytes?|[kmg]?hz|[kmg]bps|km|mi|miles?|ft|cm|mm|kg|lbs?|oz|pt|px|ms|mph|°[cf]?)|[+\-−]?\d+(?:[.,]\d+)* ?° ?[nsew]?|[+\-−]?\d+(?:[.,]\d+)* ?%|[+\-−]?(?:[$€£¥₹₩]|us\$|usd|eur|gbp) ?[+\-−]?\d+(?:[.,]\d+)*|[+\-−]?\d+(?:[.,]\d+)* ?(?:[$€£¥₹₩]|usd|eur|gbp)|(?:\d{2,}(?:[.,]\d+)?|\d+[.,]\d+) ?k|[+\-−]?\d+(?:[.,]\d+)*\+?|\([+\-−]?\d+(?:[.,]\d+)*(?: ?(?:[kmgt]i?b|bytes?|[kmg]?hz|[kmg]bps|km|mi|miles?|ft|cm|mm|kg|lbs?|oz|pt|px|ms|mph|°[cf]?))?\)|\[[+\-−]?\d+(?:[.,]\d+)*\]|\((?:mon|tues?|wed(?:nes)?|thu(?:rs?)?|fri|sat(?:ur)?|sun)(?:day)?\.?\)))*(?: today)?
```

## Style assignment

Each selected row gets one style, by id:

```python
p = int(hashlib.sha256(("style:" + row["id"]).encode("utf-8")).hexdigest(), 16) % 100
style = "functional" if p < 50 else "descriptive" if p < 80 else "positional"
```

- **functional** (~50%): an imperative action ("open more options"); for a text or area control only a neutral verb plus the label ("tap Digital Wellbeing").
- **descriptive** (~30%): a noun phrase naming the element ("the Settings button").
- **positional** (~20%): locates the element by a plain-word position and/or a neighbor relation, keeping a word of its own label ("the Share button at the top right", "the Save button below Title").

## What Lightning sees

Lightning is text-only and **never sees the screenshot**. Each target is one compact JSON line built only from manifest metadata (optional fields are left out when unknown):

- `k`: item number within the request (1..N); the reply is matched back by k.
- `label`: the raw manifest instruction (accessibility name, HTML text or link text) with invisible format characters removed (Unicode category Cf: bidi marks such as U+200E/U+2068, zero-width characters) and whitespace collapsed. Labels longer than 200 characters are cut there and end in `…`.
- `platform`: `mobile`, `web` or `desktop`, from the manifest.
- `control`: a plain noun for the row's `role` (else `data_type`), via the table below. The raw value is reduced to a key: last dotted segment, a macOS `AX` prefix dropped, lowercased, non-alphanumerics removed (`android.widget.ImageButton` -> `imagebutton`, `AXPopUpButton` -> `popupbutton`). Unknown keys become `element`; omitted when both are absent or only `text`/`icon` (those go to `kind`).
- `kind`: `text` or `icon` when the source labels it (ScreenSpot-style); omitted otherwise.
- `position`: 3x3 grid cell (thirds of width and height) holding the bbox center: `top-left`, `top-center`, `top-right`, `middle-left`, `center`, `middle-right`, `bottom-left`, `bottom-center`, `bottom-right`. A center exactly on a 1/3 line goes to the later cell. The prompt asks for plain words ("at the top right"); a copied grid token is a validation error.
- `size`: bbox area / screenshot area: `small` < 0.005, `medium` < 0.03, else `large`.
- `neighbors`: up to 8 other manifest targets on the same screenshot, nearest bbox center first (ties by label, id). Each has its label (normalized like `label`, first 60 characters) and `target_is`: where the TARGET lies relative to that neighbor, phrased to be copied (`right of` Search -> "right of Search"): `overlapping` if either center lies inside the other box, else `left of`/`right of`/`above`/`below` by the dominant axis of the center offset (ties go horizontal).
- `style`: the style to write, assigned by id (below).

`control` nouns (`meta.CONTROL_NOUNS`; `text` and `area` mark the label or region itself, never a button):

| `control` sent | role/data_type keys |
| --- | --- |
| `button` | `pushbutton`, `button`, `imagebutton`, `splitbutton` |
| `toggle button` | `togglebutton` |
| `radio button` | `radiobutton` |
| `menu button` | `menubutton` |
| `link` | `link`, `hyperlink` |
| `menu` | `menu` |
| `menu bar` | `menubar` |
| `menu item` | `menuitem`, `menubaritem`, `checkmenuitem`, `radiomenuitem` |
| `tree item` | `treeitem` |
| `list item` | `listitem` |
| `table cell` | `tablecell`, `cell` |
| `tab` | `pagetab`, `tab`, `tabitem`, `tabgroup`, `tabwidget` |
| `checkbox` | `checkbox` |
| `option` | `checkedtextview` |
| `switch` | `switch` |
| `toggle` | `compoundbutton` |
| `dropdown` | `combobox`, `popupbutton`, `spinner` |
| `text field` | `textfield`, `edittext`, `edit`, `entry`, `autocompletetextview` |
| `password field` | `passwordtext`, `securetextfield` |
| `search field` | `searchfield` |
| `text area` | `textarea` |
| `slider` | `slider`, `seekbar` |
| `scroll bar` | `scrollbar` |
| `spin box` | `spinbutton` |
| `image` | `image`, `imageview` |
| `icon` | `icon` |
| `text` | `statictext`, `label`, `textview`, `heading`, `paragraph` |
| `area` | `group`, `framelayout`, `linearlayout`, `relativelayout`, `constraintlayout`, `viewgroup`, `view`, `pane`, `panel`, `filler`, `canvas`, `frame`, `section`, `scrollarea`, `scrollview`, `scrollpane`, `recyclerview` |
| `list` | `list`, `listview` |
| `table` | `table` |
| `toolbar` | `toolbar` |
| `status bar` | `statusbar` |
| `radio group` | `radiogroup` |
| `color picker` | `colorwell` |
| `tab list` | `pagetablist`, `tablist` |
| `element` | any other key |

Neighbors are other *sampled* targets of the screenshot (the manifest keeps at most ~8 per screenshot), not every element on screen. Example user message for a one-item request, built from a synthetic screenshot by the same code (`meta.row_meta`, `prompts.format_batch`), exactly as sent:

```text
Rewrite this item. Reply with one JSON object {"items":[...]} holding exactly 1 item, k = 1 to 1, in order. Follow each item's style and keep a word of its own label. A neighbor's target_is is where the target is relative to that neighbor: copy it as is. Never invent colors, shapes, text or element types.

{"k":1,"label":"More options","platform":"mobile","control":"button","position":"top right","size":"small","neighbors":[{"label":"Search","target_is":"right of"},{"label":"Inbox","target_is":"above"}],"style":"functional"}
```

Each request is: the system prompt, the few-shot pair below, then one user message with N item lines.

## Prompt (verbatim)

### System prompt

```text
You write click instructions for training a GUI grounding model. Each item is one on-screen element, the target, with its raw accessibility or HTML label. Turn the label into a short, natural instruction for clicking it, like the ScreenSpot benchmark ("open the settings", "the Recent tab"). You cannot see the screen: use only the item's fields.

Item fields:
- label: the target's raw label.
- platform: mobile, web or desktop.
- control: plain noun for the element type, if known (button, link, tab, text field, checkbox, menu item, dropdown, text, area, element...). kind: text or icon, if known.
- position: where the target's center is on a 3x3 screen grid (top left, top center, ..., center, ..., bottom right). Say it naturally: "at the top left", "on the right", "in the middle".
- size: small, medium or large, relative to the screen.
- neighbors: other labeled elements on the same screen, nearest first; not necessarily adjacent. target_is: where the target is relative to that neighbor; copy it as is: target_is right of Search -> "right of Search".
- style: the style to write.

Styles (examples are label -> instruction):
- functional: an imperative action, verb from the label's meaning: More options -> "open more options", Email address (text field) -> "enter the email address". Unsure what the click does: a neutral verb (click, tap, select). On/off state is unknown: "toggle Bluetooth", not "turn on Bluetooth".
- descriptive: a noun phrase naming the target: Settings (button) -> "the Settings button", Recent (tab) -> "the Recent tab", Battery (text) -> "the Battery text".
- positional: locate the target by its position and/or a neighbor, keeping a word of its label: "the Share button at the top right", "the Save button below Title". Prefer an exact target_is relation to "next to" or "near".

Rules:
1. 3-8 words, one line, no final period, no quotation marks. Short labels: add the control noun or a neutral verb ("click the Copy button"), not invented context.
2. Faithful: item k's instruction is about item k's own label and keeps at least one word of it, in every style. Never describe or name a neighbor instead; neighbors only help locate the target.
3. Never invent visual details (colors, icon shapes such as arrow, gear, star or magnifying glass, styling) unless the word is in the label or a neighbor label. Size words: only "small" or "large", when size says so.
4. Never invent text or purpose: every word comes from the label, a neighbor label, control, the position or a plain verb. Search -> "click Search", not a made-up query.
5. Copy proper nouns, brands, acronyms, mixed-case words and visible names (files, domains, usernames) exactly (GitHub, Wi-Fi, PDF, iPhone, output_20240911_164440).
6. Element nouns (button, link, icon, tab, field, toggle...) come only from control or the label. control text or area: the box is the text or region itself, not a button or toggle beside it: never call it a button, link, icon, tab or toggle; functional style uses only click, tap or select ("tap Digital Wellbeing"). control element or missing (web pages): no element noun, at most a neutral item, entry or option.
7. Long labels (sentences, headlines, link texts, accessibility descriptions): in any style, a short phrase naming the thing in its own key words, never the copied sentence: "Bar Chart. Showing App usage data with 7 data points." -> "the app usage bar chart".
8. Look-alikes: if a neighbor label is identical or starts with the same word ("Download App" vs "Download PDF"), add what tells the target apart: its distinct words ("download the app"), a target_is relation or its position.
9. style only says how to phrase it: never write the words functional, descriptive or positional.
10. Non-English label: keep its words verbatim inside an English instruction.

Drop an item (keep=false) only when no faithful instruction is possible, with one drop_reason:
- numeric: only numbers, sizes, prices, dates, times, percentages, counters or relative times, even with a unit or status word ("Updated 2 hrs ago · 4.5 MB").
- unintelligible: not readable words: resource or bus IDs (":1.72/StatusNotifierItem"), AutomationID values, code names (btn_submit_2), long URLs or paths, gibberish, single characters or glyphs.
- needs_screen: too generic to identify the element, and nothing in the item pins it down ("Image", "Icon", "Button", "View", "Item 3").
- duplicate_neighbor: a neighbor has the same label (ignoring case and punctuation) and no target_is relation or position tells them apart.

Do not over-drop: most items must be kept. Ordinary short labels ("Search", "Close", "Sign in", a name or a title) and long text (shorten it) are fine. Terse is not unintelligible.

Output exactly one JSON object and nothing else (no prose, no markdown):
{"items":[{"k":1,"keep":true,"instruction":"<instruction>"},{"k":2,"keep":false,"drop_reason":"<drop_reason>"}]}
One item per input k, in order. Kept items have "instruction"; dropped items have "drop_reason".
```

### Few-shot examples

**user:**

```text
Rewrite these 9 items. Reply with one JSON object {"items":[...]} holding exactly 9 items, k = 1 to 9, in order. Follow each item's style and keep a word of its own label. A neighbor's target_is is where the target is relative to that neighbor: copy it as is. Never invent colors, shapes, text or element types.

{"k":1,"label":"Digital Wellbeing","platform":"mobile","control":"text","position":"middle left","size":"small","neighbors":[{"label":"Screen time","target_is":"below"},{"label":"Bedtime mode","target_is":"above"},{"label":"Focus mode","target_is":"above"}],"style":"functional"}
{"k":2,"label":"Updated 2 hrs ago · 4.5 MB","platform":"mobile","control":"text","position":"top left","size":"small","neighbors":[{"label":"report_final.pdf","target_is":"below"},{"label":"Recent files","target_is":"below"},{"label":"Sort","target_is":"left of"}],"style":"descriptive"}
{"k":3,"label":"Search","platform":"desktop","control":"button","position":"top right","size":"small","neighbors":[{"label":"Main Menu","target_is":"left of"},{"label":"Files","target_is":"right of"},{"label":"Close","target_is":"left of"}],"style":"positional"}
{"k":4,"label":"Bar Chart. Showing App usage data with 7 data points.","platform":"mobile","control":"area","position":"center","size":"large","neighbors":[{"label":"Today","target_is":"below"},{"label":"Instagram","target_is":"above"},{"label":"Chrome","target_is":"above"}],"style":"functional"}
{"k":5,"label":":1.72/StatusNotifierItem","platform":"desktop","control":"menu","position":"top right","size":"small","neighbors":[{"label":"System","target_is":"left of"},{"label":"Firefox Web Browser","target_is":"right of"}],"style":"functional"}
{"k":6,"label":"Download Best Share Market App","platform":"web","position":"middle right","size":"small","neighbors":[{"label":"Download Stock Market Courses","target_is":"below"},{"label":"Share Market Tips","target_is":"above"},{"label":"Download Trading Guide","target_is":"above"}],"style":"descriptive"}
{"k":7,"label":"Privacy & Security","platform":"desktop","control":"text","position":"middle left","size":"small","neighbors":[{"label":"Siri & Spotlight","target_is":"below"},{"label":"Desktop & Dock","target_is":"above"},{"label":"Touch ID & Password","target_is":"above"}],"style":"descriptive"}
{"k":8,"label":"Image","platform":"web","position":"top left","size":"medium","neighbors":[{"label":"Skip to content","target_is":"below"},{"label":"Home","target_is":"left of"},{"label":"About Us","target_is":"left of"}],"style":"descriptive"}
{"k":9,"label":"Contact Us","platform":"web","position":"bottom right","size":"small","neighbors":[{"label":"Client Handbook","target_is":"right of"},{"label":"FAQs","target_is":"right of"},{"label":"Privacy Policy","target_is":"below"}],"style":"positional"}
```

**assistant:**

```text
{"items":[{"k":1,"keep":true,"instruction":"tap Digital Wellbeing"},{"k":2,"keep":false,"drop_reason":"numeric"},{"k":3,"keep":true,"instruction":"the Search button left of Main Menu"},{"k":4,"keep":true,"instruction":"tap the app usage bar chart"},{"k":5,"keep":false,"drop_reason":"unintelligible"},{"k":6,"keep":true,"instruction":"the Best Share Market App download"},{"k":7,"keep":true,"instruction":"the Privacy & Security text"},{"k":8,"keep":false,"drop_reason":"needs_screen"},{"k":9,"keep":true,"instruction":"Contact Us at the bottom right"}]}
```

## Response format

`response_format` requested: `json_object` (`json_schema` sends the schema below; `json_object` asks for any JSON object; `none` sends nothing). If the server rejects the mode with HTTP 400, the client steps down once (json_schema -> json_object -> none) and keeps the lower mode for the rest of the run. Every reply is validated locally against this shape whatever the mode:

```json
{
  "type": "object",
  "properties": {
    "items": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "k": {
            "type": "integer"
          },
          "keep": {
            "type": "boolean"
          },
          "instruction": {
            "type": [
              "string",
              "null"
            ]
          },
          "drop_reason": {
            "type": [
              "string",
              "null"
            ],
            "enum": [
              "numeric",
              "unintelligible",
              "needs_screen",
              "duplicate_neighbor",
              null
            ]
          }
        },
        "required": [
          "k",
          "keep"
        ]
      }
    }
  },
  "required": [
    "items"
  ]
}
```

## Validation, retries and drops

- Parsing tolerates `<think>` blocks, markdown fences and prose around the outermost JSON object or array; items are matched to inputs by `k` (first item wins; unknown `k` ignored). Any style the model echoes is ignored: the style comes from the id.
- `keep=true`: `instruction` must be one line of 2-12 words, whitespace-normalized, trailing period removed, and pass every check below.
- `keep=false` must give a `drop_reason` in `numeric`, `unintelligible`, `needs_screen`, `duplicate_neighbor` (common synonyms are mapped onto these).
- Retries: one request carries up to 20 rows. Rows whose item is missing or invalid, or every row when the reply is not valid JSON or was cut off (`finish_reason=length`), are re-asked up to 2 more times (3 attempts per invocation): together, in one request per failed reply (its failed rows only, renumbered from k=1; `--group-retries`). Rows still invalid are dropped as `drop_reason=llm_error`. Requests were paced client-side at 36.0 per minute (`--rpm`).
- Attempt accounting: every failed attempt is stored per row, so a row's `attempts` counts all its LLM attempts across invocations (an abort mid-retry, `--retry-errors`), and a row that ever failed is never counted as valid on the first try.
- Output breaker: replies with no valid item extend a streak (a single-row retry whose one item failed only extends a running streak); any valid item resets it. While a streak runs, rows that use up their attempts are held back instead of becoming `llm_error`. After 5 such replies in a row the run stops with exit code 2 and those rows stay pending.
- Transport failures (HTTP 408/429/5xx, timeouts, connection errors, undecodable bodies) are retried with exponential backoff and full jitter, honoring `Retry-After`. If they persist, the rows are not cached and are re-sent on the next run. Other HTTP 4xx errors stop the run.
- `apply.py` removes selected rows with `keep=false` from both training variants, so the ablation compares identical rows.

Validation errors (the attempt fails and the row is re-asked):

| error | meaning |
| --- | --- |
| `invalid_json` | the reply had no parseable JSON object or array; every row of the request is re-asked |
| `truncated` | the reply was cut off (`finish_reason=length`); every row of the request is re-asked |
| `missing` | the reply has no item with this `k` |
| `not_object` | the item is not a JSON object |
| `bad_keep` | `keep` is not true or false |
| `bad_drop_reason` | `keep=false` without a known `drop_reason` |
| `missing_instruction` | `keep=true` without a non-empty `instruction` |
| `newline` | the instruction spans more than one line |
| `word_count:<n>` | fewer than 2 or more than 12 words |
| `invented_quote` | quotes text that is not in the label or a neighbor label |
| `invented_color:<word>` | a color word (list below) not in the label, a neighbor label or the control |
| `invented_shape:<word>` | an icon-shape word (list below) not in the label, a neighbor label or the control |
| `no_label_word` | no content word of the row's own label, in any style (catches an answer shifted to another item); labels without ASCII words are exempt |
| `wrong_direction:<neighbor>` | "left of/right of/above/below/under/beneath <neighbor>" contradicts that neighbor's `target_is` ("next to", "near", "between" are not checked) |
| `invented_control:<noun>` | an element-type noun (text field, check box, drop-down, drop down, button, icon, link, tab, field, input, checkbox, dropdown, menu, slider, switch, toggle, image, picture; input/switch/toggle as the first word are verbs) not backed by the control, the label or a neighbor label: without a control none at all, except button or link on web pages (generic clickable nouns), and for a text/area control only words such as text, label, area, item, entry, option |
| `text_action_verb:<verb>` | a functional instruction for a text/area control that starts with a verb other than choose, click, find, go, press, select, tap (no "open X" on a text) |
| `style_word:<word>` | the word functional, descriptive or positional, unless a label has it |
| `grid_token:<token>` | a hyphenated grid token (top-left, middle-right, ...) copied from `position` |
| `ambiguous_lookalike:<neighbor>` | a neighbor label is identical or starts with the same word, and the instruction has none of the target's distinguishing words, no relation to a neighbor and no position word (identical labels need a relation or position) |

Warnings (counted, the row is kept):

| warning | meaning |
| --- | --- |
| `stripped_style_word` | a style name (functional/descriptive/positional) copied at the very start or end of the instruction was removed deterministically (validate.cleanup_instruction) |
| `dehyphenated_grid` | a hyphenated grid token (top-right) was rewritten as plain words (top right) |
| `unchanged` | the instruction equals the label |
| `length` | outside the targeted 3-8 words |
| `neighbor_swap:<label>` | a functional/descriptive instruction names a neighbor's label and no word of the target's label |
| `proper_noun:<token>` | a label token with an internal capital, digits and letters, or all caps is missing |

- Color words: black, blue, brown, cyan, golden, gray, green, grey, magenta, orange, pink, purple, red, silver, teal, violet, white, yellow.
- Shape words: arrow, bell, checkmark, chevron, circle, circular, cog, gear, hamburger, heart, kebab, magnifier, magnifying, oval, pencil, plus, round, rounded, square, star, tick, trash, triangle, triangular.
- Before any request the manifest load aborts if a row's source, id or image path mentions ScreenSpot or GroundUI, so no ScreenSpot rows are in the inputs (asserted). ScreenSpot-v2-clean is the held-out test. OS-Atlas Linux/macOS desktop data is training data; the 28 ScreenSpot-v2 desktop rows captured on the same machines are excluded from the test set instead (`benchmarks/screenspot_v2/excluded_rows.json`).

## Results

Status: complete. 9384 rows rewritten or dropped: 8620 kept, 764 dropped (353 of them by the numeric rule, never sent).

By source:

| source | targets | kept | dropped | of which by rule | drop rate |
| --- | --- | --- | --- | --- | --- |
| os_atlas/androidworld | 1694 | 1411 | 283 | 185 | 16.7% |
| os_atlas/fineweb | 3920 | 3759 | 161 | 91 | 4.1% |
| os_atlas/linux | 1819 | 1699 | 120 | 2 | 6.6% |
| os_atlas/macos | 1951 | 1751 | 200 | 75 | 10.2% |

By style:

| style | targets | kept |
| --- | --- | --- |
| functional | 4693 | 4272 |
| descriptive | 2784 | 2601 |
| positional | 1907 | 1747 |

Drop reasons (`numeric` includes the rule drops):

| drop_reason | rows |
| --- | --- |
| duplicate_neighbor | 26 |
| llm_error | 24 |
| needs_screen | 29 |
| numeric | 421 |
| unintelligible | 264 |

Validation warnings on kept rows (counted, not dropped):

| warning | count |
| --- | --- |
| length | 883 |
| neighbor_swap | 1 |
| proper_noun | 84 |
| stripped_style_word | 423 |
| unchanged | 185 |

Validation errors on failed attempts (those rows were re-asked):

| error | count |
| --- | --- |
| ambiguous_lookalike | 153 |
| bad_drop_reason | 1 |
| invented_control | 199 |
| missing | 1 |
| no_label_word | 176 |
| style_word | 151 |
| text_action_verb | 465 |
| word_count | 75 |
| wrong_direction | 196 |

Quality gate (the faithfulness check is manual, on the `review_sample.py` contact sheet):

| metric | value | gate | pass |
| --- | --- | --- | --- |
| valid on first try (of 9031 rows sent to the model; rule drops excluded) | 89.5% | >= 95% | no |
| drop rate, all drops except llm_error (rule drops included) | 7.9% | < 25% | yes |
| drop rate of the model (rows sent; rule drops excluded) | 4.3% |  | n/a |
| rows dropped by the numeric rule (never sent) | 353 |  | n/a |
| batch replies that parsed as JSON | 100.0% |  | n/a |
| rows that needed more than one attempt (across invocations) | 946 |  | n/a |
| rows dropped as llm_error | 24 |  | n/a |
| faithful on manual review | see review.html | >= 90% | n/a |

Requests and tokens (`all runs` sums every invocation for this prompt version on this cache):

|  | this run | all runs |
| --- | --- | --- |
| requests | 0 | 1015 |
| HTTP attempts | 0 | 1932 |
| HTTP retries | 0 | 917 |
| prompt tokens | 0 | 3330622 |
| completion tokens | 0 | 214016 |
| total tokens | 0 | 3544638 |
| latency p50 (s) | n/a | 59.596 |
| latency p95 (s) | n/a | 268.053 |
| by status | {} | {"ok": 375, "partial": 576, "transport_error": 64} |
| by kind | {} | {"batch": 526, "retry": 259, "single": 230} |

Throughput over all 5 invocations that sent at least one request (sum of rows / sum of wall time; an invocation that sends nothing, such as a resume, does not change it):

| invocations | rows | requests | wall time (s) | rows/min | requests/min | completion tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 9495 | 1015 | 5983.966 | 95.2 | 10.18 | 35.76 |

Those invocations:

| run id | started / finished (UTC) | wall (s) | rows | rule drops | requests | prompt / completion tokens | endpoint | model label | served checkpoint | command |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `20261017T204115Z-ef71ce` | 2026-10-17T20:41:15Z / 2026-10-17T20:48:23Z | 428.567 | 194 | 6 | 67 | 177642 / 5227 | hosted (`https://integrate.api.nvidia.com/v1`) | `NVIDIA-hosted API, nvidia/nemotron-3.5-lightning-30b-a3b` | not reported | `python data_prep/rewrite/rewrite.py --manifest data/os_atlas_final/manifest.jsonl --out data/os_atlas_final/rewrite_v4 --endpoint hosted --concurrency 16 --splits train --limit 200 --note 'Generated 2026-10-17 (US Eastern evening) via the NVIDIA-hosted API (build.nvidia.com), model nvidia/nemotron-3.5-lightning-30b-a3b. The agent and demo use the official local checkpoint nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 (vllm-metal); the prompt was developed with dry runs on both.'` |
| `20261017T205047Z-c39514` | 2026-10-17T20:50:47Z / 2026-10-17T20:51:19Z | 31.827 | 78 | 347 | 12 | 19752 / 1601 | hosted (`https://integrate.api.nvidia.com/v1`) | `NVIDIA-hosted API, nvidia/nemotron-3.5-lightning-30b-a3b` | not reported | `python data_prep/rewrite/rewrite.py --manifest data/os_atlas_final/manifest.jsonl --out data/os_atlas_final/rewrite_v4 --endpoint hosted --concurrency 40 --splits train --retry-errors --note 'Generated 2026-10-17 (US Eastern evening) via the NVIDIA-hosted API (build.nvidia.com), model nvidia/nemotron-3.5-lightning-30b-a3b. The agent and demo use the official local checkpoint nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 (vllm-metal); the prompt was developed with dry runs on both.'` |
| `20261017T205532Z-d98d23` | 2026-10-17T20:55:32Z / 2026-10-17T21:18:55Z | 1402.513 | 4020 | 0 | 411 | 1459164 / 90348 | hosted (`https://integrate.api.nvidia.com/v1`) | `NVIDIA-hosted API, nvidia/nemotron-3.5-lightning-30b-a3b` | not reported | `python data_prep/rewrite/rewrite.py --manifest data/os_atlas_final/manifest.jsonl --out data/os_atlas_final/rewrite_v4 --endpoint hosted --concurrency 24 --rpm 36 --max-attempts 10 --group-retries --splits train --retry-errors --note 'Generated 2026-10-17 (US Eastern evening) via the NVIDIA-hosted API (build.nvidia.com), model nvidia/nemotron-3.5-lightning-30b-a3b. The agent and demo use the official local checkpoint nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 (vllm-metal); the prompt was developed with dry runs on both.'` |
| `20261017T212032Z-9fadcf` | 2026-10-17T21:20:32Z / 2026-10-17T22:16:43Z | 3370.411 | 3899 | 0 | 405 | 1249056 / 86976 | hosted (`https://integrate.api.nvidia.com/v1`) | `NVIDIA-hosted API, nvidia/nemotron-3.5-lightning-30b-a3b` | not reported | `python data_prep/rewrite/rewrite.py --manifest data/os_atlas_final/manifest.jsonl --out data/os_atlas_final/rewrite_v4 --endpoint hosted --concurrency 24 --rpm 36 --max-attempts 10 --group-retries --max-consecutive-empty-replies 10 --splits train --retry-errors --note 'Generated 2026-10-17 (US Eastern evening) via the NVIDIA-hosted API (build.nvidia.com), model nvidia/nemotron-3.5-lightning-30b-a3b. The agent and demo use the official local checkpoint nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 (vllm-metal); the prompt was developed with dry runs on both.'` |
| `20261017T220349Z-a4bffb` | 2026-10-17T22:03:49Z / 2026-10-17T22:16:19Z | 750.648 | 1304 | 0 | 120 | 425008 / 29864 | local (`http://127.0.0.1:8000/v1`) | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | `python data_prep/rewrite/rewrite.py --manifest data/os_atlas_final/manifest.jsonl --out data/os_atlas_final/rewrite_v4 --endpoint local --concurrency 3 --group-retries --max-consecutive-empty-replies 10 --splits train --retry-errors --reverse --note 'Generated 2026-10-17 (US Eastern evening) by two endpoints serving NVIDIA Nemotron 3.5 Lightning 30B-A3B: the NVIDIA-hosted API (build.nvidia.com, model nvidia/nemotron-3.5-lightning-30b-a3b) and, after hosted rate-limiting at ~18:00, the official local checkpoint nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 (vllm-metal on a Mac, also used by the agent and demo). Each row'"'"'s endpoint and served checkpoint are in the cache; counts are in the provenance table.'` |

## Reproduce

```bash
python data_prep/rewrite/rewrite.py --manifest data/os_atlas_final/manifest.jsonl --out data/os_atlas_final/rewrite_v4 --endpoint hosted --splits train --group-retries --rpm 36 --note 'Generated 2026-10-17 (US Eastern evening) by two endpoints serving NVIDIA Nemotron 3.5 Lightning 30B-A3B: the NVIDIA-hosted API (build.nvidia.com, model nvidia/nemotron-3.5-lightning-30b-a3b) and, after hosted rate-limiting at ~18:00, the official local checkpoint nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 (vllm-metal on a Mac, also used by the agent and demo). Each row'"'"'s endpoint and served checkpoint are in the cache; counts are in the provenance table.'
```

|  |  |
| --- | --- |
| git commit | `d0d9d625436be1ce6eef9184206330df9da0c1a1` |
| manifest | `/Users/pballou/projects/nvidia-hack-26/.claude/worktrees/t3-data-engine/data/os_atlas_final/manifest.jsonl` |
| manifest sha256 | `4ceb7ea3d7ac8e850fde81eaa935ed6d9d88de45a22a1faf007bc1d2cce541fa` |
| manifest rows | 19621 |
| rows by source | {"os_atlas/androidworld": 3000, "os_atlas/fineweb": 6986, "os_atlas/linux": 3271, "os_atlas/macos": 3364, "os_atlas/uibert": 3000} |
| selected by source | {"os_atlas/androidworld": 1694, "os_atlas/fineweb": 3920, "os_atlas/linux": 1819, "os_atlas/macos": 1951} |
| targets this run | 9384 |
| run id | `20261017T221703Z-ee3acf` |
| started / finished (UTC) | 2026-10-17T22:17:03Z / 2026-10-17T22:17:04Z |

Sampling at temperature 0.7 is not deterministic, so `rewrites.jsonl` is the artifact. Rerunning the command against the same `cache.sqlite` (keyed by prompt_version and id) sends no requests and rewrites `rewrites.jsonl` byte for byte; `apply.py` is deterministic given it. The commands of the invocations that sent requests are in the table above.

## rewrites.jsonl fields

| field | meaning |
| --- | --- |
| `id` | manifest row id (`<subset>:<screenshot stem>:<k>`), which names the OS-Atlas screenshot |
| `source` | manifest source, e.g. `os_atlas/androidworld` |
| `original` | the raw label exactly as in the manifest |
| `rewritten` | Lightning's instruction after validation, or null when dropped |
| `style` | requested style (functional / descriptive / positional) |
| `keep` | false if the row was dropped by rule, by Lightning, or failed validation 3 times |
| `drop_reason` | null, one of numeric, unintelligible, needs_screen, duplicate_neighbor, or llm_error |
| `prompt_version` | prompt version (cache key together with id) |
| `model` | model id that produced the row, or `rule:is_numeric_label` for numeric labels dropped before any request |
| `ts` | UTC time the row was generated |

## Known limitations

- Lightning works from metadata only. A label that is ambiguous without the pixels can still produce an instruction that fits the label but not the boxed element. The `needs_screen` and `duplicate_neighbor` drops and the manual review reduce this but do not remove it.
- Some source boxes are on the wrong element (a label boxed on a neighboring button); the rewrite follows the label, so such rows stay wrong. The manual review counts them separately (source error), not as rewrite errors.
- Position is a coarse 3x3 bucket and size is three buckets, so positional instructions are approximate.
- Neighbors are only the targets sampled from the same screenshot (at most ~8), so "between A and B" may skip unlabeled elements in between.
- Lints are word lists and heuristics: they catch invented colors, icon shapes, quoted text, element nouns, contradicted directions and shifted answers, but not every hallucination. Faithfulness was audited by hand on a random sample (review_sample.py), not on every row.
- The hosted endpoint's weights and serving stack are controlled by NVIDIA and may change between runs; `models_reported` records what the server said.
- Numeric rule drops remove those rows from both training variants, which shifts the training mix slightly away from numeric labels. Model drops and `llm_error` rows are kept with their raw label (see the audit section).

## Human audit and how drops are applied

A person reviewed a random sheet from this run (`review_sample.py`, seed 0, run `20261017T204115Z-ef71ce`; verdicts in `review_verdicts_v4_seed0.json`): each card shows the screenshot crop with the target box, the raw label and the rewrite.

|  | cards |
| --- | --- |
| rewrites judged faithful | 42 |
| rewrites judged unfaithful | 1 |
| unsure (could not be verified from the crop: hidden text, look-alikes, non-English) | 6 |
| source error (box is not on the labeled element; not a rewrite error) | 0 |
| not judged | 1 |

Faithful: 42/43 = 97.7% of the cards that could be verified (gate: >= 90%), or 42/49 = 85.7% counting unsure cards as not faithful.

Model drops: 3 of 16 sampled drops were judged correct and 13 wrong; readable labels and file names are valid click targets. Overturned examples: "0Likes" (unintelligible), "3.9.19 ('osworld': conda), ~/anaconda3/e" (numeric), "output_20240912_151942" (numeric), "PSP plus NINTENDO" (unintelligible), "chor" (unintelligible), "No apps can perform this action." (unintelligible), "Spaces: 4" (unintelligible), "Darknet Cannabis Markets" (unintelligible), "accessibility_visualization.png" (unintelligible), "GitLens is" (unintelligible).

How `apply.py` uses drops: only the deterministic numeric rule removes rows. A row Lightning dropped (`keep=false`, any `drop_reason`) or that ended as `llm_error` stays in both training files with its raw label, `instruction_style` `raw` and the reason in `lightning_drop_reason`.

## Data sources and license

- **OS-Atlas** ([OS-Copilot/OS-Atlas-data](https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data), Apache-2.0 per the dataset card). Z. Wu et al., "OS-ATLAS: A Foundation Action Model for Generalist GUI Agents", arXiv:2410.23218, 2024. Labels, boxes and screenshots come from it.
- Screenshots are **not re-hosted**: `id` and the manifest's `image_path` refer to the OS-Atlas files by name.
- Rewrites were generated by `NVIDIA-hosted API, nvidia/nemotron-3.5-lightning-30b-a3b` (8028 rows), `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (1003 rows) (NVIDIA Nemotron 3.5 Lightning); 353 numeric labels were dropped by rule, without a model.
- No ScreenSpot (v1/v2) or GroundUI-18K data is used (asserted at load time). ScreenSpot-v2-clean (revision `5efbb1f1b5` minus the 28 desktop rows in `benchmarks/screenspot_v2/excluded_rows.json`) is the held-out test.
