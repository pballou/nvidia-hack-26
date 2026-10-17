"""validate.py: tolerant JSON parsing, batch keying, item normalization, the numeric label rule and the faithfulness
lints (v3 neighbors: {"label", "target_is"} = where the target lies relative to that neighbor)."""

import json

import pytest

from data_prep.rewrite import validate as v


def meta(label="Settings", neighbors=("Search", "Red alert"), control="button", style="functional",
         target_is="right of", **extra):
    """Neighbors are labels (all at `target_is`) or (label, target_is) pairs; control is a plain noun or None."""
    pairs = [(n, target_is) if isinstance(n, str) else n for n in neighbors]
    m = {"label": label, "platform": "mobile", "position": "top-right", "size": "small",
         "neighbors": [{"label": lab, "target_is": rel} for lab, rel in pairs], "style": style, **extra}
    if control:
        m["control"] = control
    return m


def item(k=1, instruction="open the settings", keep=True, **kw):
    return {"k": k, "keep": keep, "instruction": instruction, **kw}


def check(instruction, m=None):
    """validate_item for one keep=True rewrite -> (rewritten or None, error, warnings)."""
    normalized, error, warnings = v.validate_item(item(instruction=instruction), m or meta())
    return (normalized or {}).get("rewritten"), error, warnings


def err(instruction, m=None):
    return check(instruction, m)[1]


GOOD = json.dumps({"items": [item(1, "open the settings."), {"k": 2, "keep": False, "drop_reason": "numeric"},
                             item(3, "the Search button")]})
METAS3 = [meta(), meta(label="42"), meta(label="Search", neighbors=("Settings",))]


# --- payload parsing -----------------------------------------------------------------------------------------------

def test_valid_batch():
    ok, errors, warnings, payload_ok = v.validate_batch(GOOD, METAS3)
    assert payload_ok and errors == {} and warnings == {}
    assert ok == {1: {"keep": True, "rewritten": "open the settings", "drop_reason": None},
                  2: {"keep": False, "rewritten": None, "drop_reason": "numeric"},
                  3: {"keep": True, "rewritten": "the Search button", "drop_reason": None}}


@pytest.mark.parametrize("text", [
    f"```json\n{GOOD}\n```",
    f"Here are the rewrites:\n```\n{GOOD}\n```\nLet me know if you need more.",
    f"<think>Row 2 is a number {{so drop}} it.</think>\n{GOOD}",
    f"<THINK>\nreasoning\n</THINK>```json\n{GOOD}```",
    f"reasoning without an opening tag</think>{GOOD}",
    f"Sure! {GOOD} Hope this helps [1].",
    f"Note [see format]: {GOOD}",
    json.dumps(json.loads(GOOD)["items"]),
    json.dumps(json.loads(GOOD), indent=2),
])
def test_tolerant_payloads(text):
    ok, errors, _, payload_ok = v.validate_batch(text, METAS3)
    assert payload_ok and errors == {} and sorted(ok) == [1, 2, 3]


def test_parse_json_payload_returns_object():
    assert v.parse_json_payload('<think>{"x": 1}</think> {"items": []}') == {"items": []}
    assert v.parse_json_payload('prefix [1, 2] suffix') == [1, 2]
    assert v.parse_json_payload('{"a": "line\nbreak"}') == {"a": "line\nbreak"}


@pytest.mark.parametrize("text", [
    "", "   ", "not json at all", "<think>only thoughts</think>", GOOD[:-20], '{"items": [{"k": 1, "keep": tru',
    '{"result": "ok"}', '{"items": "nope"}', "42", '"a string"', "```json\n```",
])
def test_malformed_payload(text):
    ok, errors, warnings, payload_ok = v.validate_batch(text, METAS3)
    assert not payload_ok and ok == {} and warnings == {}
    assert sorted(errors) == [1, 2, 3] and all(e.startswith("invalid_json: ") for e in errors.values())


def test_parse_rejects_non_text():
    with pytest.raises(ValueError):
        v.parse_json_payload(None)
    with pytest.raises(ValueError):
        v.parse_json_payload("no json here")


def test_extract_items():
    assert v.extract_items({"items": [1]}) == [1]
    assert v.extract_items([{"k": 1}]) == [{"k": 1}]
    single = {"k": 1, "keep": False, "drop_reason": "numeric"}
    assert v.extract_items(single) == [single]
    for bad in ({"items": {"k": 1}}, {"rows": []}, "x", 3, None):
        with pytest.raises(ValueError):
            v.extract_items(bad)


# --- batch keying --------------------------------------------------------------------------------------------------

def test_missing_item():
    ok, errors, _, payload_ok = v.validate_batch(json.dumps({"items": [item(1)]}), METAS3[:2])
    assert payload_ok and list(ok) == [1] and errors == {2: "missing"}


def test_duplicate_k_first_wins():
    text = json.dumps({"items": [item(1, "open the settings"), item(1, "tap the settings entry")]})
    ok, errors, _, _ = v.validate_batch(text, [meta()])
    assert ok[1]["rewritten"] == "open the settings" and errors == {}
    text = json.dumps({"items": [item(1, "settings"), item(1, "open the settings")]})
    ok, errors, _, _ = v.validate_batch(text, [meta()])
    assert ok == {} and errors == {1: "word_count:1"}


def test_string_float_and_bad_k():
    items = [item("2", "open search"), item(1.0), item(True, "tap something"), item(0), item(9), {"keep": True},
             "junk", None]
    ok, errors, _, payload_ok = v.validate_batch(json.dumps(items), [meta(), METAS3[2], meta()])
    assert payload_ok and sorted(ok) == [1, 2] and errors == {3: "missing"}
    assert ok[2]["rewritten"] == "open search"


def test_errors_and_warnings_keyed_by_k():
    items = [item(1, "settings"), item(2, "open GitHub settings now please")]
    ok, errors, warnings, _ = v.validate_batch(json.dumps(items), [meta(), meta(label="GitHub settings")])
    assert errors == {1: "word_count:1"} and list(ok) == [2] and 2 not in warnings


def test_batch_uses_each_items_own_meta():
    """The shifted-items failure: every item got the next item's instruction; each is checked against its own label."""
    labels = ["Others", "Entertainment", "Groceries"]
    metas = [meta(label=lab, control="text", neighbors=()) for lab in labels]
    shifted = json.dumps({"items": [item(1, "click Entertainment"), item(2, "click Groceries"),
                                    item(3, "click Others")]})
    ok, errors, _, _ = v.validate_batch(shifted, metas)
    assert ok == {} and errors == {1: "no_label_word", 2: "no_label_word", 3: "no_label_word"}
    aligned = json.dumps({"items": [item(k, f"click {lab}") for k, lab in enumerate(labels, 1)]})
    ok, errors, _, _ = v.validate_batch(aligned, metas)
    assert errors == {} and sorted(ok) == [1, 2, 3]


# --- item normalization --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("keep,expected", [
    (True, True), (False, False), ("true", True), ("FALSE", False), (" True ", True),
    ("yes", None), (1, None), (0, None), (None, None), ("", None),
])
def test_keep_types(keep, expected):
    normalized, error, _ = v.validate_item(item(keep=keep, drop_reason="numeric"), meta())
    if expected is None:
        assert normalized is None and error == "bad_keep"
    else:
        assert error is None and normalized["keep"] is expected


@pytest.mark.parametrize("reason,expected", [
    ("numeric", "numeric"), ("unintelligible", "unintelligible"), ("needs_screen", "needs_screen"),
    ("duplicate_neighbor", "duplicate_neighbor"), ("needs_visual", "needs_screen"), ("ambiguous", "needs_screen"),
    ("duplicate", "duplicate_neighbor"), ("nonsense", "unintelligible"), ("Needs Screen", "needs_screen"),
    ("needs-visual", "needs_screen"), ("NUMERIC ", "numeric"),
])
def test_drop_reason_aliases(reason, expected):
    normalized, error, warnings = v.validate_item({"k": 1, "keep": False, "drop_reason": reason}, meta())
    assert error is None and warnings == []
    assert normalized == {"keep": False, "rewritten": None, "drop_reason": expected}


@pytest.mark.parametrize("reason", ["llm_error", "other", "", None, 3, "too_long"])
def test_bad_drop_reason(reason):
    assert v.validate_item({"k": 1, "keep": False, "drop_reason": reason}, meta()) == (None, "bad_drop_reason", [])


def test_all_aliases_target_known_reasons():
    assert set(v.DROP_REASON_ALIASES.values()) <= set(v.DROP_REASONS)
    assert v.LLM_ERROR not in v.DROP_REASONS and v.STYLES == ("functional", "descriptive", "positional")


def test_model_style_ignored_and_shape():
    normalized, error, _ = v.validate_item(item(style="positional", drop_reason="numeric"), meta())
    assert error is None and normalized == {"keep": True, "rewritten": "open the settings", "drop_reason": None}


@pytest.mark.parametrize("raw,expected", [
    ("  open   the\tsettings  ", "open the settings"),
    ("open the settings.", "open the settings"),
    ("open the settings...", "open the settings"),
    ('"open the settings"', "open the settings"),
    ("open the settings\n", "open the settings"),
])
def test_instruction_normalization(raw, expected):
    assert check(raw)[:2] == (expected, None)


@pytest.mark.parametrize("raw", [None, "", "   ", 5, ["open settings"], "."])
def test_missing_instruction(raw):
    assert check(raw)[1] == "missing_instruction"


def test_rewritten_key_fallback():
    normalized, error, _ = v.validate_item({"k": 1, "keep": True, "rewritten": "open the settings"}, meta())
    assert error is None and normalized["rewritten"] == "open the settings"


def test_newline_rejected():
    assert check("open the\nsettings")[1] == "newline"


def test_not_object():
    assert v.validate_item("open settings", meta()) == (None, "not_object", [])


# --- numeric label rule (applied before Lightning) -----------------------------------------------------------------

@pytest.mark.parametrize("label", [
    # the contract's must-match list
    "42", "1,234", "$4.99", "€12", "45%", "12:45 PM", "10:00", "Oct 12", "13 Oct 2023 10:00 AM", "2023-10-13",
    "10/13/2023", "73 minutes ago", "5 min ago", "2h ago", "3 days ago", "1 hour ago", "now", "just now", "+3", "(12)",
    # seen in the OS-Atlas 20k selection
    "0 min. ago", "01:00 PM - 01:30 PM", "1 hr, 10 min today", "55 minutes today", "14 Saturday", "October 13 (Fri)",
    "1:06:00 (3.26 mi)", "161K", "772.3K", "52.29 s", "48kHz", "37.64 MiB", "18.72 GB", "0.35 pt",
    "2020-12-11T21:33:33-06:00", "OCT 15 - 22", "2023 Oct 1", "15. August 2017", "Oct '22", "+8h", "1w", "15ms",
    "4h 48m", "00h 00m 09s", "Today, September 5", "Today, 6:22 PM", "Yesterday 8:00 AM", "Sunday, October 8",
    "Tuesday 3rd September, 2024", "AUGUST 1, 2018 AT 7:50 AM", "Nov 18, 2023, 12:42 PM", "Dec 1, 2022", "June 2024",
    "3 min read", "32 minutes", "2,645 of 4,188", "3 of 10", "37.37593° N, 122.08395° W", "-$4.99", "12+", "[3]",
    "a few seconds ago", "an hour ago", "in 5 min", "5 min left",
    # normalization: case, bidi marks, narrow no-break space (macOS clock), trailing period
    "‎42‎", "12:45 PM", "JUST NOW", "Oct 12.",
])
def test_numeric_labels(label):
    assert v.is_numeric_label(label)


@pytest.mark.parametrize("label", [
    # the contract's must-not list
    "Today", "Yesterday", "Start Wednesday, October 4", "Wi-Fi 6", "iPhone 15 Pro", "Windows 11", "Top 10 tips",
    "3D Touch", "4K", "Page 2", "Item 3", "COVID-19", "24/7 Support",
    # the few-shot's borderline numeric item reaches the model; words around numbers keep a label
    "Updated 2 hrs ago · 4.5 MB", "8K", "3D", "5G", "PS4", "A1", "X1", "Now Playing", "May", "Sat", "Sunday",
    "0 Comments", "Week before due 09:39", "Last changed 3 months ago", "Rated 4 out of 5", "Psalm 110", "v0.113.1",
    "Episode 1", "Pac-12", "Search", "", "   ", "now playing 3", "#F57C00",
])
def test_non_numeric_labels(label):
    assert not v.is_numeric_label(label)


def test_numeric_rule_is_bounded():
    assert not v.is_numeric_label("1 " * 40)                                   # past NUMERIC_LABEL_MAX_CHARS
    assert v.is_numeric_label("1 " * 20) and not v.is_numeric_label("5 m " * 15 + "x")
    assert v.NUMERIC_LABEL_RE.pattern and v.NUMERIC_LABEL_RE.flags & 2         # re.I, published verbatim


# --- lints: word count, quotes, colors, shapes ---------------------------------------------------------------------

@pytest.mark.parametrize("n,error,length_warning", [
    (1, "word_count:1", True), (2, None, True), (3, None, False), (8, None, False), (9, None, True),
    (12, None, True), (13, "word_count:13", True),
])
def test_word_count_bounds(n, error, length_warning):
    text = " ".join(["open", "settings", "the"] + ["now"] * 20)
    rewritten, e, warnings = check(" ".join(text.split()[:n]))
    assert e == error and ("length" in warnings) is length_warning
    assert (rewritten is None) is (error is not None)


@pytest.mark.parametrize("text,error", [
    ("tap the blue settings icon", "invented_color:blue"),
    ("tap the GREY settings icon", "invented_color:grey"),
    ("open the settings gear", "invented_shape:gear"),
    ("tap the magnifying glass", "invented_shape:magnifying"),
    ("tap the three stars", "invented_shape:star"),
    ("tap the circular settings button", "invented_shape:circular"),
    ("tap the rounded settings button", "invented_shape:rounded"),
    ("tap the red and blue settings", "invented_color:blue"),   # red is in a neighbor label; blue is not
    ("open the settings around here", None),                    # word boundaries: around != round
    ("start the settings wizard", None),                        # start != star
    ("redo the settings change", None),                         # redo != red
])
def test_invented_color_shape(text, error):
    assert err(text) == error


def test_color_shape_allowed_from_metadata():
    assert err("tap the blue theme button", meta(label="Blue theme")) is None
    assert err("tap the settings button by red alert", meta()) is None         # neighbor label has "Red"
    assert err("tap the star next to search", meta(label="Stars rating")) is None   # simple plural
    assert err("tap the stars next to search", meta(label="Star")) is None
    assert err("tap the settings arrow at the top", meta(control="arrow")) is None   # from control
    assert err("tap the settings arrow at the top", meta(control=None)) == "invented_shape:arrow"
    assert err("open the pink settings", meta(neighbors=("Pink Floyd",))) is None


@pytest.mark.parametrize("text,error", [
    ("type in the 'Username' field", "invented_quote"),
    ('tap "Sign out" in the menu', "invented_quote"),
    ("tap “Log in” at the top", "invented_quote"),
    ("tap ‘Log in’ at the top", "invented_quote"),
    ("tap 'settings' at the top", None),                 # case-insensitive substring of the label
    ('tap the settings button right of "Search"', None),  # neighbor label
    ("tap 'Red' settings near the top", None),           # substring of neighbor "Red alert"
    ("open the user's settings page", None),             # apostrophe, not a quote
    ("don't open the settings", None),
])
def test_invented_quote(text, error):
    assert err(text) == error


def test_unchanged_warning():
    rewritten, error, warnings = check("open settings.", meta(label="Open settings"))
    assert error is None and rewritten == "open settings" and "unchanged" in warnings
    assert "unchanged" not in check("open the settings page")[2]


@pytest.mark.parametrize("label,text,warning", [
    ("GitHub repo", "open the repo page", "proper_noun:GitHub"),
    ("iPhone 15 Pro", "view the Pro phone details", "proper_noun:iPhone"),
    ("MP3 player", "open the music player", "proper_noun:MP3"),
    ("USB-C cable", "buy the charging cable", "proper_noun:USB"),
    ("AutomationID:x", "open the automation panel", "proper_noun:AutomationID"),
])
def test_proper_noun_warning(label, text, warning):
    rewritten, error, warnings = check(text, meta(label=label))
    assert error is None and rewritten == text and warning in warnings


@pytest.mark.parametrize("label,text", [
    ("GitHub repo", "open the GitHub repo"),
    ("GitHub repo", "open the github repo"),
    ("USB-C cable", "buy the usb-c cable"),
    ("Settings", "open the settings page"),
    ("42 items", "show all 42 items"),
])
def test_no_proper_noun_warning(label, text):
    assert not [w for w in check(text, meta(label=label))[2] if w.startswith("proper_noun")]


def test_warnings_returned_with_valid_item_only_in_batch():
    text = json.dumps({"items": [item(1, "open the repo page")]})
    ok, errors, warnings, _ = v.validate_batch(text, [meta(label="GitHub repo")])
    assert list(ok) == [1] and errors == {} and warnings == {1: ["proper_noun:GitHub"]}


@pytest.mark.parametrize("label,neighbors,style,text,error,warning", [
    # naming a neighbor and no word of the target's label is now an error too (no_label_word); the warning stays
    ("Camera", ("More options", "Gmail"), "functional", "open more options", "no_label_word",
     "neighbor_swap:More options"),
    ("iPhone 15 Pro", ("USB-C cable",), "descriptive", "read about USB-C cable", "no_label_word",
     "neighbor_swap:USB-C cable"),
    ("Camera", ("More options",), "positional", "the camera image right of More options", None, None),
    ("Camera", ("More options",), "functional", "open the camera options", None, None),   # shares a label word
    ("Navigate up", ("Search",), "functional", "go back", "no_label_word", None),       # synonyms no longer pass
    ("Share", ("Share", "Menu"), "functional", "open the share sheet at the top", None, None),
    ("Image", ("Image",), "descriptive", "the large image at the top", None, None),     # same label as the target
    ("Inbox", ("Search",), "functional", "open the research inbox", None, None),
    ("Back", ("Search",), "functional", "start a research query", "no_label_word", None),  # "search" inside a word
])
def test_neighbor_swap_warning(label, neighbors, style, text, error, warning):
    rewritten, e, warnings = check(text, meta(label=label, neighbors=neighbors, style=style, control="image"))
    assert e == error and (rewritten == text) is (error is None)
    assert [w for w in warnings if w.startswith("neighbor_swap")] == ([warning] if warning else [])


# --- no_label_word: every rewrite keeps a word of its own label ----------------------------------------------------

@pytest.mark.parametrize("label,text,style", [
    ("Others", "click Entertainment", "functional"),                       # the v2 shifted batch
    ("Environmental", "open Coordinates search", "functional"),
    ("accounts.google.com", "open 13 Oct 2023 10:00 AM", "descriptive"),
    ("Donation", "the Environmental entry on the left", "positional"),
    ("Search", "the button at the top right", "positional"),               # positional keeps a label word too
    ("Navigate up", "go back to the previous page", "functional"),
    ("Más opciones", "open more options", "functional"),                   # non-English words stay verbatim
    ("Log out", "sign off", "functional"),
    ("Entertainment", "click Enter", "functional"),                        # a stem shared by chance is not enough
    ("Page 12", "go to the next one", "functional"),
])
def test_no_label_word_errors(label, text, style):
    assert err(text, meta(label=label, style=style, neighbors=(), control="text")) == "no_label_word"


@pytest.mark.parametrize("label,text", [
    ("Others", "click Others"),
    ("Settings", "click the setting"),                       # inflection
    ("Library", "tap the libraries"),
    ("GitHub repo", "click the repo"),
    ("Log out", "tap Logout"),                               # adjacent label words joined
    ("Logout", "tap log out"),
    ("Wi-Fi", "tap WiFi"),
    ("Más opciones", "click Más opciones"),
    ("Inicia una conversación", "tap Inicia una conversación"),
    ("设置", "tap the settings"),                             # no ASCII content word: skipped
    ("…", "tap the ellipsis"),
    ("Bar Chart. Showing App usage data with 7 data points.", "the app usage bar chart"),
    ("X1", "tap X1 at the top"),
    ("OUT CROPPINGS: IMPORTANT CROP NEWS FROM THE FIELD!", "tap the crop news headline"),
])
def test_no_label_word_ok(label, text):
    assert err(text, meta(label=label, neighbors=(), control="text")) is None


# --- wrong_direction: relation phrases must agree with target_is ---------------------------------------------------

def nb_meta(*neighbors, label="Terminal", control="button", style="positional"):
    return meta(label=label, neighbors=neighbors, control=control, style=style)


@pytest.mark.parametrize("text,neighbors,error", [
    # v2's inverted cases: the target is RIGHT of Explorer, the model said left
    ("the Terminal button left of Explorer Section: osworld", (("Explorer Section: osworld", "right of"),),
     "wrong_direction:Explorer Section: osworld"),
    ("the Terminal button right of Explorer Section: osworld", (("Explorer Section: osworld", "right of"),), None),
    ("the Terminal button left of Explorer Section", (("Explorer Section: osworld", "right of"),),   # >= 12 chars
     "wrong_direction:Explorer Section: osworld"),
    ("the Terminal button left of Explorer", (("Explorer Section: osworld", "right of"),), None),    # too short
    ("the Terminal button above Search", (("Search", "below"),), "wrong_direction:Search"),
    ("the Terminal button below Search", (("Search", "below"),), None),
    ("the Terminal button under Search", (("Search", "below"),), None),         # under/beneath == below
    ("the Terminal button beneath the Search", (("Search", "above"),), "wrong_direction:Search"),
    ("the Terminal button right of the Search", (("Search", "left of"),), "wrong_direction:Search"),
    ("the Terminal button to the left of SEARCH", (("Search", "left of"),), None),   # case-insensitive
    ('the Terminal button left of "Search"', (("Search", "right of"),), "wrong_direction:Search"),
    ("the Terminal button above Mark as Read", (("Mark as Read", "overlapping"),), "wrong_direction:Mark as Read"),
    ("the Terminal button left of Help", (("Help", "left of"), ("Help", "below")), None),  # any same-label agrees
    ("the Terminal button above Help", (("Help", "left of"), ("Help", "right of")), "wrong_direction:Help"),
    ("the Terminal button left of Homepage", (("Home", "right of"),), None),     # "Home" is not named
    ("the Terminal button left of Home", (("Home", "right of"), ("Homepage", "left of")), "wrong_direction:Home"),
    ("the Terminal button left of Home page", (("Home", "right of"), ("Home page", "left of")), None),  # longest wins
    ("the Terminal button right of Thesaurus", (("Thesaurus...", "left of"),), "wrong_direction:Thesaurus..."),
    ("the Terminal text below The early bird", (("The early bird catches the worm.", "above"),),
     "wrong_direction:The early bird catches the worm."),
    ("the Terminal button next to Search", (("Search", "below"),), None),       # unchecked relations
    ("the Terminal button near Search", (("Search", "below"),), None),
    ("the Terminal button between Search and Help", (("Search", "below"), ("Help", "above")), None),
    ("the Terminal button left of the dock", (("Search", "below"),), None),     # not a neighbor
    ("the Terminal button at the top left", (("Search", "below"),), None),
])
def test_wrong_direction(text, neighbors, error):
    assert err(text, nb_meta(*neighbors)) == error


def test_wrong_direction_v2_cases():
    m = nb_meta(("Search", "below"), ("Collections", "right of"), label="Microsoft Shopping")
    assert err("Microsoft Shopping right of Search", m) == "wrong_direction:Search"
    assert err("Microsoft Shopping below Search", m) is None
    m = nb_meta(("Home", "right of"), ("Subscribe", "above"), label="OUT CROPPINGS: IMPORTANT CROP NEWS", control=None)
    assert err("the crop news headline left of Home", m) == "wrong_direction:Home"
    assert err("the crop news headline right of Home", m) is None


# --- invented_control: element nouns only from the metadata --------------------------------------------------------

@pytest.mark.parametrize("text,control,label,error", [
    # control text/area: the box is the text itself
    ("the 73 minutes ago button", "text", "73 minutes ago", "invented_control:button"),
    ("the Time text field at the top", "text", "Time", "invented_control:text field"),
    ("the Meeting with the Team text field", "text", "Meeting with the Team", "invented_control:text field"),
    ("the Digital Wellbeing toggle", "text", "Digital Wellbeing", "invented_control:toggle"),
    ("the Shopify chat icon", "area", "Shopify chat", "invented_control:icon"),
    ("the Digital Wellbeing text", "text", "Digital Wellbeing", None),
    ("the Shopify chat area", "area", "Shopify chat", None),
    ("the text labeled Battery", "text", "Battery", None),
    ("the Battery label entry", "text", "Battery", None),
    # control missing (FineWeb): no element noun at all
    ("the Podcast button", None, "Podcast", "invented_control:button"),
    ("the Home link at the top left", None, "Home", "invented_control:link"),
    ("the Pin on Pinterest icon", None, "Pin on Pinterest", "invented_control:icon"),
    ("the Contact tab at the top", None, "Contact", "invented_control:tab"),
    ("the Openings text field", None, "Openings", "invented_control:text field"),
    ("the Share on LinkedIn link", None, "Share on LinkedIn", "invented_control:link"),   # LinkedIn != link
    ("the Openings menu", None, "Openings", "invented_control:menu"),
    ("the Openings dropdown", None, "Openings", "invented_control:dropdown"),
    ("the Openings drop-down", None, "Openings", "invented_control:drop-down"),
    ("the Openings images", None, "Openings", "invented_control:image"),                 # plural
    ("the Podcast option at the top", None, "Podcast", None),
    ("the Cleaner Handbook at the center", None, "Cleaner Handbook", None),
    ("the Search Button", None, "Search Button", None),                                   # from the label
    ("open in another tab", None, "Resource will open in another tab", None),
    # control nouns support their own words and aliases
    ("the Search button", "button", "Search", None),
    ("the Search toggle button", "toggle button", "Search", None),
    ("the Title field", "text field", "Title", None),
    ("the Title input", "text field", "Title", None),
    ("the Title text field", "text field", "Title", None),
    ("the File menu", "menu item", "File", None),
    ("the Bluetooth toggle", "switch", "Bluetooth", None),
    ("the Bluetooth switch", "toggle", "Bluetooth", None),
    ("the Remember me check box", "checkbox", "Remember me", None),
    ("the Avatar picture", "image", "Avatar", None),
    ("the Search link", "button", "Search", "invented_control:link"),
    ("the Bluetooth toggle", "button", "Bluetooth", "invented_control:toggle"),
    ("the Search button", "element", "Search", "invented_control:button"),
    ("the Search button", "table cell", "Search", "invented_control:button"),
    ("the Recent tab", "tab list", "Recent", None),
    # first-word verbs are not nouns
    ("toggle Bluetooth", "button", "Bluetooth", None),
    ("switch to Recent", "button", "Recent", None),
    ("toggle the Bluetooth toggle", "button", "Bluetooth", "invented_control:toggle"),
])
def test_invented_control(text, control, label, error):
    assert err(text, meta(label=label, control=control, neighbors=(), style="descriptive")) == error


def test_invented_control_supported_by_neighbor_or_kind():
    m = meta(label="Settings", control=None, neighbors=(("Main Menu", "below"),), style="descriptive")
    assert err("the Settings menu", m) is None
    assert err("the Settings button", m) == "invented_control:button"
    assert err("the Settings icon", meta(label="Settings", control=None, neighbors=(), kind="icon")) is None


# --- text_action_verb: text/area elements take a neutral click verb ------------------------------------------------

@pytest.mark.parametrize("text,control,style,error", [
    ("open Digital Wellbeing", "text", "functional", "text_action_verb:open"),
    ("Open Digital Wellbeing", "text", "functional", "text_action_verb:open"),
    ("show Bar Chart of Digital Wellbeing", "area", "functional", "text_action_verb:show"),
    ("enable Digital Wellbeing", "area", "functional", "text_action_verb:enable"),
    ("search for Digital Wellbeing", "text", "functional", "text_action_verb:search"),
    ("toggle Digital Wellbeing", "text", "functional", "text_action_verb:toggle"),
    ("tap Digital Wellbeing", "text", "functional", None),
    ("click the Digital Wellbeing text", "text", "functional", None),
    ("select Digital Wellbeing", "area", "functional", None),
    ("go to Digital Wellbeing", "text", "functional", None),
    ("press Digital Wellbeing", "text", "functional", None),
    ("the Digital Wellbeing text", "text", "functional", None),   # a noun phrase is not an action verb
    ("open Digital Wellbeing", "button", "functional", None),     # buttons may open things
    ("open Digital Wellbeing", None, "functional", None),
])
def test_text_action_verb(text, control, style, error):
    assert err(text, meta(label="Digital Wellbeing", control=control, style=style, neighbors=())) == error


# --- style_word / grid_token: no prompt vocabulary in the output ---------------------------------------------------

@pytest.mark.parametrize("text,label,error", [
    ("the Screen Time text positional", "Screen Time", "style_word:positional"),
    ("the note text descriptive", "note", "style_word:descriptive"),
    ("the wuwei text Functional", "wuwei", "style_word:functional"),
    ("open the Functional tests", "Functional tests", None),     # in the label
    ("the Screen Time text at the top", "Screen Time", None),
])
def test_style_word(text, label, error):
    assert err(text, meta(label=label, control="text", style="descriptive", neighbors=())) == error


@pytest.mark.parametrize("text,error", [
    ("the Maximize Panel Size at the middle-right", "grid_token:middle-right"),
    ("the Move To Home button bottom-left", "grid_token:bottom-left"),
    ("the Maximize Panel Size top–center", "grid_token:top–center"),
    ("the Maximize Panel Size at the middle right", None),
    ("the Maximize Panel Size on the right", None),
    ("the Maximize Panel Size top center", None),
    ("the Maximize Panel Size in the top right corner", None),
])
def test_grid_token(text, error):
    label = "Move To Home" if "Move" in text else "Maximize Panel Size"
    assert err(text, meta(label=label, control="button", style="positional", neighbors=())) == error


def test_grid_token_in_label_is_fine():
    assert err("open the top-left corner", meta(label="Top-left corner", control="button", neighbors=())) is None


# --- ambiguous_lookalike: look-alike neighbors must be told apart --------------------------------------------------

@pytest.mark.parametrize("label,neighbors,text,error", [
    # same first content word: the instruction needs a word the neighbor lacks, a relation or a position
    ("Download PDF", ("Download now",), "click the Download button", "ambiguous_lookalike:Download now"),
    ("Download PDF", ("Download now",), "download the PDF", None),
    ("Download Best Share Market App", ("Download now",), "click Download Best Share Market App", None),
    ("Download PDF", ("Download now",), "click Download at the top", None),
    ("Download PDF", (("Download now", "left of"),), "click Download left of Download now", None),
    ("Download PDF", (("Download now", "left of"),), "click Download next to Download now", None),
    # identical labels need a relation or a position
    ("Home", ("Home",), "click Home", "ambiguous_lookalike:Home"),
    ("CONTACT", ("Contact",), "click CONTACT", "ambiguous_lookalike:Contact"),
    ("Home", ("Home",), "click Home at the top center", None),
    ("Home", (("Home", "left of"), ("Blog", "below")), "click Home below Blog", None),
    ("Top stories", ("Top stories",), "open Top stories", "ambiguous_lookalike:Top stories"),  # "top" is the label
    ("Top stories", ("Top stories",), "open Top stories on the left", None),
    # the whole label quoted tells them apart unless the neighbor contains it
    ("Sign in", ("Sign up",), "click Sign in", None),
    ("Rated 5 out of 5", ("Rated 4 out of 5",), "click Rated 5 out of 5", None),
    ("yyyy/MM/dd", ("yyyy-MM-dd",), "click yyyy/MM/dd", None),
    ("Settings", ("Settings Sync, group",), "click Settings", "ambiguous_lookalike:Settings Sync, group"),
    ("Settings", ("Settings Sync, group",), "click Settings at the top", None),
    # single characters and non-ASCII words count
    ("X share", ("Share this story on LinkedIn",), "click X share", None),
    ("关于 Google", ("Google 搜索的运作方式",), "click 关于 Google", None),
    ("Page 2", ("Page 3",), "go to page 2", None),
    ("Page 2", ("Page 3",), "go to the page", "ambiguous_lookalike:Page 3"),
    # not look-alikes
    ("Android chips", ("Android Vcard library",), "tap Android chips", None),
    ("Search", ("Settings",), "click Search", None),
    ("Offline Wikipedia available as part of OsmAnd", (("Offline Wikipedia", "overlapping"),),
     "tap Offline Wikipedia", "ambiguous_lookalike:Offline Wikipedia"),
])
def test_ambiguous_lookalike(label, neighbors, text, error):
    assert err(text, meta(label=label, neighbors=neighbors, control="button")) == error


# --- check order -----------------------------------------------------------------------------------------------------

def test_first_failing_check_is_reported():
    m = meta(label="OUT CROPPINGS: IMPORTANT CROP NEWS FROM THE FIELD!", control=None, style="positional",
             neighbors=(("Home", "right of"),))
    assert err("the large text field top-center left of Home", m) == "wrong_direction:Home"
    assert v.CHECKS[0].__name__ == "_no_label_word"
    errors = [c("the large text field top-center left of Home", m) for c in v.CHECKS]
    assert [e for e in errors if e] == ["wrong_direction:Home", "grid_token:top-center"]   # "field" is in the label
    assert err("the Donation entry top-center", meta(label="Donation", control="text", neighbors=())) \
        == "grid_token:top-center"
